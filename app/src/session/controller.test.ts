import type Anthropic from "@anthropic-ai/sdk";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MOCK_DEFAULT_SETTINGS, MockBackend } from "../backend/mock";
import type { ScriptLine } from "../backend/mockScript";
import { SUGGEST_SYSTEM_PROMPT, SUGGESTION_SCHEMA } from "../llm/suggestions";
import { fakeClaude, MemoryStorage, requestText, taskOf, waitFor, type CapturedRequest } from "../test/helpers";
import { SessionController } from "./controller";

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
});
afterEach(() => {
  vi.unstubAllGlobals();
});

const SCRIPT: ScriptLine[] = [
  { who: "aigerim", pause: 20, text: "коллеги давайте начнем синк" },
  // Named in calling position but not a question: Claude decides it wasn't for the user.
  { who: "dina", pause: 20, text: "хуго вчера скидывал сценарии возвратов" },
  // Right after: the SKIP must not count against the debounce.
  { who: "aigerim", pause: 20, text: "хугоға сұрақ возвраты қашан дайын болады" },
  { who: "me", pause: 20, text: "в среду закроем" },
  { who: "erlan", pause: 20, text: "всем привет я по продукту" },
  { who: "erlan", pause: 20, text: "релиз в пятницу нас устраивает", stray: true },
];

const HINT = "Скажите: **в среду** закроем возвраты.";

function reply(req: CapturedRequest): string {
  if (req.system[0].text === SUGGEST_SYSTEM_PROMPT) {
    return JSON.stringify({ suggestions: [{ label: "sys:2", name: "Дина", evidence: "«хуго вчера скидывал…» — Дина из QA", confidence: 0.8 }] });
  }
  const task = taskOf(req);
  if (task.includes('<task type="hint">')) return task.includes("скидывал") ? "SKIP" : HINT;
  if (task.includes('<task type="final_summary">')) return "## Итоги\nВсё решили.";
  return "ок";
}

async function setup() {
  const backend = new MockBackend({ speed: 50, script: SCRIPT });
  await backend.saveSettings({ ...MOCK_DEFAULT_SETTINGS, autoHintMode: "addressed", rollingSummaryMinutes: 0 });
  await backend.setApiKey("sk-ant-test");
  const claude = fakeClaude(reply);
  const controller = new SessionController({ backend, createClient: () => claude.client });
  await controller.init();
  return { backend, controller, calls: claude.calls };
}

const isHint = (r: CapturedRequest) => taskOf(r).includes('<task type="hint">');

describe("SessionController with the mock backend and a fake Claude", () => {
  it("runs a multi-party session: SKIP, rename notes, suggestions, relabel, final summary", async () => {
    const { backend, controller, calls } = await setup();
    const state = () => controller.getState();
    expect(state().settings.myNames).toEqual(["Хуго"]);

    await controller.start("Синк", "повестка");
    await waitFor(() => state().finals.length === SCRIPT.length, 15_000, "all finals");
    await waitFor(() => calls.filter(isHint).length === 2 && state().cards.every((c) => c.status === "done"), 5_000, "two auto hints");

    // Both addressed questions fired; the first was SKIP and left no trace.
    const hints = calls.filter(isHint);
    expect(taskOf(hints[0])).toContain("ответь ровно SKIP");
    expect(taskOf(hints[0])).toContain("Участник 2: «хуго вчера скидывал сценарии возвратов»");
    expect(taskOf(hints[1])).toContain("хугоға сұрақ");
    expect(state().cards).toHaveLength(1);
    expect(state().cards[0]).toMatchObject({ label: "Обращаются к вам", text: HINT, hidden: false, auto: true });
    expect(state().cards[0].detail).toBe("Участник 1: хугоға сұрақ возвраты қашан дайын болады");
    const meetingId = state().meetingId!;
    const notes = (await backend.getMeeting(meetingId)).notes;
    expect(notes.filter((n) => n.kind === "hint").map((n) => n.content)).toEqual([HINT]);

    // Online meeting: mic = «me», call voices = sys:N (sys:4 is the spurious cluster).
    expect(state().finals.map((f) => f.speaker)).toEqual(["sys:1", "sys:2", "sys:1", "me", "sys:3", "sys:4"]);

    // Suggestions: a separate structured-output request off the main conversation.
    await controller.suggestNames();
    const side = calls.at(-1)!;
    expect(side.system).toEqual([{ type: "text", text: SUGGEST_SYSTEM_PROMPT }]);
    expect(side.messages).toHaveLength(1);
    expect(side.output_config).toMatchObject({ effort: "low", format: SUGGESTION_SCHEMA });
    expect(JSON.stringify(side)).not.toContain("cache_control");
    expect(state().suggestions).toHaveLength(1);
    expect(state().suggestions[0]).toMatchObject({ label: "sys:2", name: "Дина" });

    // Accepting renames everywhere at once and persists through rename_speaker.
    controller.acceptSuggestion(state().suggestions[0].id);
    await controller.renameSpeaker("sys:3", "Ерлан");
    expect(state().speakerNames).toEqual({ "sys:2": "Дина", "sys:3": "Ерлан" });
    expect(state().suggestions).toEqual([]);
    const stored = await backend.getMeeting(meetingId);
    expect(stored.speakers.find((s) => s.label === "sys:2")?.name).toBe("Дина");

    // The next live turn: earlier turns untouched, a rename note, the names map in the task.
    controller.hint();
    await waitFor(() => calls.filter(isHint).length === 3, 5_000, "manual hint");
    const third = calls.filter(isHint)[2];
    const second = hints[1];
    const strip = (r: CapturedRequest) => JSON.stringify(r.messages.map((m) => (typeof m.content === "string" ? m.content : m.content.map((b) => b.text))));
    expect(strip(third).startsWith(strip(second).slice(0, -2))).toBe(true);
    const tail = third.messages.at(-1)!.content as Array<{ text: string }>;
    expect(tail[0].text).toBe("<speaker_names>\nУчастник 2 теперь зовут Дина.\n</speaker_names>");
    expect(tail[1].text).toContain("Ерлан: всем привет я по продукту");
    expect(taskOf(third)).toContain("Имена участников: Участник 2 — Дина; Участник 3 — Ерлан.");
    expect(taskOf(third)).not.toContain("SKIP");

    // Stop: the mock re-clusters (sys:4 → sys:3); the final summary sees the fixed labels.
    await controller.stop();
    await waitFor(() => state().final.status === "done", 5_000, "final summary");
    expect(state().finals.at(-1)!.speaker).toBe("sys:3");
    const final = calls.find((r) => taskOf(r).includes("final_summary"))!;
    expect(requestText(final)).toContain("Ерлан: релиз в пятницу нас устраивает");
    expect(requestText(final)).toContain("Дина: хуго вчера скидывал сценарии возвратов");
    expect(taskOf(final)).toContain("## Участники и позиции");
    controller.dispose();
  });

  it("(b) an unnamed question followed by silence asks Claude to decide; SKIP leaves no card", async () => {
    const backend = new MockBackend({ speed: 50, script: [{ who: "erlan", pause: 20, text: "коллеги кто может взять ревью макетов" }] });
    await backend.saveSettings({ ...MOCK_DEFAULT_SETTINGS, autoHintMode: "addressed", rollingSummaryMinutes: 0 });
    await backend.setApiKey("sk-ant-test");
    const claude = fakeClaude(() => "SKIP");
    const controller = new SessionController({ backend, createClient: () => claude.client });
    await controller.init();
    await controller.start("Синк", "");
    await waitFor(() => claude.calls.length === 1, 6_000, "the silence hint");
    const task = taskOf(claude.calls[0]);
    expect(task).toContain("все замолчали");
    expect(task).toContain("Меня по имени не называли");
    expect(task).toContain("ответь ровно SKIP");
    await waitFor(() => controller.getState().cards.length === 0 && claude.calls.length === 1);
    await controller.stop();
    controller.dispose();
  }, 10_000);

  it("(b) speech during the wait cancels it: someone else answered", async () => {
    const script: ScriptLine[] = [
      { who: "erlan", pause: 20, text: "коллеги кто может взять ревью макетов" },
      { who: "aidos", pause: 20, text: "я возьму" },
    ];
    const backend = new MockBackend({ speed: 50, script });
    await backend.saveSettings({ ...MOCK_DEFAULT_SETTINGS, autoHintMode: "addressed", rollingSummaryMinutes: 0 });
    await backend.setApiKey("sk-ant-test");
    const claude = fakeClaude(() => "SKIP");
    const controller = new SessionController({ backend, createClient: () => claude.client });
    await controller.init();
    await controller.start("Синк", "");
    await waitFor(() => controller.getState().finals.length === 2, 10_000, "finals");
    await new Promise((r) => setTimeout(r, 2_700));
    expect(claude.calls).toHaveLength(0);
    await controller.stop();
    controller.dispose();
  }, 10_000);

  it("mode \"any\" never fires on «me» and never allows SKIP", async () => {
    const script: ScriptLine[] = [
      { who: "me", pause: 20, text: "а когда у нас релиз" },
      { who: "erlan", pause: 20, text: "а когда у нас релиз" },
    ];
    const backend = new MockBackend({ speed: 50, script });
    await backend.saveSettings({ ...MOCK_DEFAULT_SETTINGS, autoHintMode: "any", rollingSummaryMinutes: 0 });
    await backend.setApiKey("sk-ant-test");
    const claude = fakeClaude(() => "SKIP");
    const controller = new SessionController({ backend, createClient: () => claude.client });
    await controller.init();
    await controller.start("Синк", "");
    await waitFor(() => claude.calls.length === 1 && controller.getState().cards[0]?.status === "done", 5_000, "one hint");
    expect(taskOf(claude.calls[0])).toContain("Участник 1: «а когда у нас релиз»");
    expect(taskOf(claude.calls[0])).not.toContain("SKIP");
    // Not skippable, so even a literal "SKIP" is shown.
    expect(controller.getState().cards[0]).toMatchObject({ hidden: false, text: "SKIP" });
    await controller.stop();
    controller.dispose();
  });
});

describe("SessionController: races", () => {
  const LINES: ScriptLine[] = [
    { who: "aigerim", pause: 20, text: "коллеги давайте начнем синк" },
    { who: "erlan", pause: 20, text: "релиз в пятницу нас устраивает" },
    { who: "dina", pause: 20, text: "тесты готовы к четвергу" },
  ];

  it("a rolling summary still streaming is never used as «the summary»: the final summary gets the last complete one", async () => {
    const backend = new MockBackend({ speed: 50, script: LINES });
    // Haiku's 200K window, so the calibrated estimate below pushes the final summary onto the summary fallback.
    await backend.saveSettings({ ...MOCK_DEFAULT_SETTINGS, claudeModel: "claude-haiku-4-5", autoHintMode: "off", rollingSummaryMinutes: 0 });
    await backend.setApiKey("sk-ant-test");
    const calls: CapturedRequest[] = [];
    let rolling = 0;
    const client = {
      beta: {
        messages: {
          stream(params: CapturedRequest, opts: { signal?: AbortSignal }) {
            calls.push(params);
            const task = taskOf(params);
            const handlers: Array<(d: string) => void> = [];
            const message = (text: string, input_tokens = 10) => ({
              model: params.model,
              stop_reason: "end_turn",
              stop_details: null,
              content: [{ type: "text", text }],
              usage: { input_tokens, output_tokens: 5, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 },
            });
            return {
              on(event: string, fn: (d: string) => void) {
                if (event === "text") handlers.push(fn);
                return this;
              },
              async finalMessage() {
                await new Promise((r) => setTimeout(r, 1));
                if (task.includes('<task type="rolling_summary">') && ++rolling === 1) {
                  handlers.forEach((h) => h("ПОЛНОЕ РЕЗЮМЕ"));
                  // Says the meeting's text is huge: the final summary will need the summary fallback.
                  return message("ПОЛНОЕ РЕЗЮМЕ", 1_000_000);
                }
                if (task.includes('<task type="rolling_summary">')) {
                  // The second summary streams half a sentence and then hangs until aborted.
                  handlers.forEach((h) => h("ОБРЫВОК"));
                  return new Promise((_, reject) => opts.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))));
                }
                return message("## Итоги\nВсё решили.");
              },
            };
          },
        },
      },
    };
    const controller = new SessionController({ backend, createClient: () => client as unknown as Anthropic });
    await controller.init();
    await controller.start("Синк", "");
    await waitFor(() => controller.getState().finals.length === LINES.length, 10_000, "finals");
    await controller.generateRolling();
    expect(controller.getState().rolling).toMatchObject({ status: "done", text: "ПОЛНОЕ РЕЗЮМЕ" });
    void controller.generateRolling();
    await waitFor(() => controller.getState().rolling.text === "ОБРЫВОК", 5_000, "the second summary to start streaming");

    await controller.stop();
    await waitFor(() => controller.getState().final.status === "done", 5_000, "final summary");
    const final = calls.find((r) => taskOf(r).includes('<task type="final_summary">'))!;
    expect(requestText(final)).not.toContain("<full_transcript>");
    expect(requestText(final)).toContain("<summary_so_far>\nПОЛНОЕ РЕЗЮМЕ\n</summary_so_far>");
    expect(requestText(final)).not.toContain("ОБРЫВОК");
    controller.dispose();
  }, 20_000);

  it("Stop while start_session is still in flight: the session that starts afterwards is stopped too", async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    class SlowStart extends MockBackend {
      // Like the Rust shell: stop_session before start_session has registered the session is a no-op.
      override async startSession(title: string, context: string) {
        await gate;
        return super.startSession(title, context);
      }
    }
    const backend = new SlowStart({ speed: 50, script: LINES });
    await backend.saveSettings({ ...MOCK_DEFAULT_SETTINGS, rollingSummaryMinutes: 0 });
    const controller = new SessionController({ backend, createClient: () => fakeClaude(() => "ок").client });
    await controller.init();
    const starting = controller.start("Синк", "");
    await waitFor(() => controller.getState().phase === "starting", 2_000, "starting");
    await controller.stop();
    expect(controller.getState().phase).toBe("stopped");
    release();
    await starting;
    expect(await backend.sessionStatus()).toBeNull();
    expect(controller.getState().phase).toBe("stopped");
    controller.dispose();
  }, 10_000);
});

describe("SessionController: reattach after a webview reload", () => {
  const SCRIPT_R: ScriptLine[] = [
    { who: "aigerim", pause: 20, text: "коллеги давайте начнем синк" },
    { who: "aigerim", pause: 20, text: "хугоға сұрақ возвраты қашан дайын болады" },
    { who: "me", pause: 20, text: "в среду закроем" },
    { who: "erlan", pause: 20, text: "всем привет я по продукту" },
    // The page reloads during this pause.
    { who: "erlan", pause: 40_000, text: "релиз в пятницу нас устраивает" },
    { who: "dina", pause: 20, text: "тесты готовы к четвергу" },
  ];
  const ROLLING = "**Главное:** релиз в пятницу.";

  function replyR(req: CapturedRequest): string {
    const task = taskOf(req);
    if (task.includes('<task type="hint">')) return HINT;
    if (task.includes('<task type="rolling_summary">')) return ROLLING;
    if (task.includes('<task type="recap">')) return "Кратко: всё по плану.";
    if (task.includes('<task type="final_summary">')) return "## Итоги\nВсё решили.";
    return "ок";
  }

  it("restores transcript, names, hints, rolling summary, timer, usage and the Claude conversation; keeps listening", async () => {
    const settings = { ...MOCK_DEFAULT_SETTINGS, autoHintMode: "addressed" as const, rollingSummaryMinutes: 0 };
    const backendA = new MockBackend({ speed: 50, script: SCRIPT_R });
    await backendA.saveSettings(settings);
    await backendA.setApiKey("sk-ant-test");
    const claude = fakeClaude(replyR);
    const a = new SessionController({ backend: backendA, createClient: () => claude.client });
    await a.init();
    await a.start("Синк", "повестка");
    await waitFor(() => a.getState().finals.length === 4 && a.getState().cards[0]?.status === "done", 10_000, "four finals and the auto hint");
    await a.renameSpeaker("sys:2", "Ерлан");
    await a.generateRolling();
    const before = a.getState();
    expect(before.rolling.text).toBe(ROLLING);
    expect(before.usage.requests).toBe(2);

    // Reload: the page (backend adapter + controller) goes away, the session keeps running.
    backendA.detach();
    a.dispose();
    const backendB = new MockBackend({ speed: 50, script: SCRIPT_R });
    const b = new SessionController({ backend: backendB, createClient: () => claude.client });
    await b.init();
    const st = b.getState();
    expect(st.phase).toBe("running");
    expect(st.meetingId).toBe(before.meetingId);
    expect(st.title).toBe("Синк");
    expect(st.context).toBe("повестка");
    expect(Math.abs(st.startedAt! - before.startedAt!)).toBeLessThan(1_000);
    expect(st.finals.map((f) => f.id)).toEqual(before.finals.map((f) => f.id));
    expect(st.speakerNames).toEqual({ "sys:2": "Ерлан" });
    expect(st.cards).toHaveLength(1);
    expect(st.cards[0]).toMatchObject({ kind: "hint", status: "done", text: HINT, label: "Подсказка (авто)", auto: true });
    expect(st.rolling).toMatchObject({ status: "done", text: ROLLING });
    expect(st.usage.requests).toBe(2);
    expect(st.toasts.some((t) => t.text.includes("Подключились"))).toBe(true);

    // The script goes on and the controller keeps receiving it.
    await waitFor(() => b.getState().finals.length === SCRIPT_R.length, 10_000, "the rest of the script");

    // The rebuilt conversation: same frozen prefix, all stored lines in the first update, current names.
    const seen = claude.calls.length;
    const heard: string[] = [];
    b.onHotkeyAction((action) => heard.push(action));
    backendB.simulateHotkey("hint");
    await waitFor(() => claude.calls.length === seen + 1 && b.getState().cards[0]?.status === "done", 5_000, "hotkey hint");
    expect(heard).toEqual(["hint"]);
    const req = claude.calls.at(-1)!;
    const firstA = claude.calls[0];
    expect(req.system).toEqual(firstA.system);
    expect((req.messages[0].content as Array<{ text: string }>)[0].text).toBe((firstA.messages[0].content as Array<{ text: string }>)[0].text);
    expect(req.messages).toHaveLength(1);
    const update = (req.messages[0].content as Array<{ text: string }>)[1].text;
    expect(update.startsWith("<transcript_update>")).toBe(true);
    for (const line of SCRIPT_R) expect(update).toContain(line.text);
    expect(update).toContain("Ерлан: всем привет я по продукту");
    expect(taskOf(req)).toContain("Имена участников: Участник 2 — Ерлан.");
    expect(b.getState().usage.requests).toBe(3);

    backendB.simulateHotkey("recap");
    await waitFor(() => b.getState().cards[0]?.kind === "recap" && b.getState().cards[0].status === "done", 5_000, "hotkey recap");

    await b.stop();
    await waitFor(() => b.getState().final.status === "done", 5_000, "final summary");
    expect(await backendB.sessionStatus()).toBeNull();
    b.dispose();
  }, 20_000);

  it("a hotkey outside a meeting only explains itself", async () => {
    const backend = new MockBackend({ speed: 50, script: [] });
    const claude = fakeClaude(() => "x");
    const c = new SessionController({ backend, createClient: () => claude.client });
    await c.init();
    expect(c.getState().phase).toBe("idle");
    expect(c.getState().hotkeys).toMatchObject({ backend: "none" });
    backend.simulateHotkey("hint");
    expect(claude.calls).toHaveLength(0);
    expect(c.getState().toasts.at(-1)?.text).toContain("когда идёт встреча");
    c.dispose();
  });
});
