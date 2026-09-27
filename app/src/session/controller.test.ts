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
