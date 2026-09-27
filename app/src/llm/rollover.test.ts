import { describe, expect, it } from "vitest";
import { fakeClaude, requestText, type CapturedRequest } from "../test/helpers";
import type { Segment } from "../types";
import { contextWindow, payloadChars, PREPARE_FRACTION, ROLLOVER_FRACTION } from "./budget";
import { Conversation, type MessageParam, type RequestPayload, type RolloverInfo } from "./conversation";
import { Copilot } from "./copilot";
import { formatContextBlock, SYSTEM_PROMPT } from "./prompts";
import { SpeakerDirectory } from "./speakers";
import { finalSummaryTask, hintTask, rollingSummaryTask } from "./tasks";

const CONTEXT = formatContextBlock({
  title: "Долгий синк",
  context: "",
  profile: "",
  setup: { captureMic: true, captureSystem: true, micMode: "me" },
  voiceprint: false,
  myNames: ["Хуго"],
  date: "2026-09-27, воскресенье",
});

function seg(n: number, text = `реплика номер ${n} про сроки релиза`): Segment {
  const source = n % 3 === 0 ? "mic" : "system";
  return { id: `${source}-${n}`, source, speaker: source === "mic" ? "me" : `sys:${(n % 2) + 1}`, startMs: n * 5_000, endMs: n * 5_000 + 3_000, text, isFinal: true };
}

type Block = { type: string; text?: string; cache_control?: unknown };
const blocks = (m: MessageParam): Block[] => (typeof m.content === "string" ? [{ type: "text", text: m.content }] : (m.content as Block[]));

function strip(p: RequestPayload | CapturedRequest) {
  const clean = (b: Block) => ({ type: b.type, text: b.text });
  return {
    system: p.system.map((b) => clean(b as Block)),
    messages: (p.messages as MessageParam[]).map((m) => ({ role: m.role, content: blocks(m).map(clean) })),
  };
}

function isPrefix(prev: RequestPayload | CapturedRequest, next: RequestPayload | CapturedRequest): boolean {
  const a = strip(prev);
  const b = strip(next);
  if (JSON.stringify(a.system) !== JSON.stringify(b.system) || a.messages.length > b.messages.length) return false;
  return a.messages.every((m, i) => JSON.stringify(m) === JSON.stringify(b.messages[i]));
}

function breakpointCount(p: RequestPayload | CapturedRequest): number {
  let n = p.system.filter((b) => (b as Block).cache_control).length;
  for (const m of p.messages as MessageParam[]) n += blocks(m).filter((b) => b.cache_control).length;
  return n;
}

describe("budget", () => {
  it("knows the context windows: 1M for current Opus/Sonnet/Fable, 200K for Haiku 4.5 and older or unknown ids", () => {
    expect(contextWindow("claude-opus-5")).toBe(1_000_000);
    expect(contextWindow("claude-opus-5-5")).toBe(1_000_000);
    expect(contextWindow("claude-sonnet-4-6")).toBe(1_000_000);
    expect(contextWindow("claude-fable-5-1")).toBe(1_000_000);
    expect(contextWindow("claude-haiku-4-5")).toBe(200_000);
    expect(contextWindow("claude-haiku-4-5-20251001")).toBe(200_000);
    expect(contextWindow("claude-opus-4-5")).toBe(200_000);
    expect(contextWindow("claude-sonnet-4-20250514")).toBe(200_000);
    expect(contextWindow("some-proxy-model")).toBe(200_000);
  });

  it("counts the characters of every text block", () => {
    expect(payloadChars({ system: [{ text: "abc" }], messages: [{ content: [{ type: "text", text: "de" }] }, { content: "fgh" }] })).toBe(8);
  });
});

describe("Conversation.rollover", () => {
  it("starts a new log seeded with the summary, the newest lines and the names map; append-only within each log", () => {
    const speakers = new SpeakerDirectory("me", { "sys:1": "Айдос" });
    const c = new Conversation(SYSTEM_PROMPT, CONTEXT, speakers);
    const before: RequestPayload[] = [];
    let n = 0;
    for (let turn = 0; turn < 4; turn++) {
      for (let i = 0; i < 5; i++) c.addFinal(seg(++n));
      before.push(c.beginTurn(hintTask({ lang: "auto", speakers }).text));
      c.commit(`ответ ${turn}`);
    }
    for (let i = 1; i < before.length; i++) expect(isPrefix(before[i - 1], before[i])).toBe(true);

    // Two more lines are pending when the log rolls over; they go into the seed, not lost.
    c.addFinal(seg(++n));
    c.addFinal(seg(++n));
    const lineChars = 60; // a rendered line is ~45-55 chars here
    const info: RolloverInfo = c.rollover({ summary: "**Главное:** обсуждаем релиз.", maxTranscriptChars: lineChars * 6 });
    expect(info).toMatchObject({ epoch: 1, summary: true });
    expect(info.carriedLines).toBeGreaterThanOrEqual(5);
    expect(info.carriedLines + info.droppedLines).toBe(n);
    expect(c.epoch).toBe(1);
    expect(c.exchanges).toBe(0);
    expect(c.pendingCount()).toBe(0);

    const after: RequestPayload[] = [];
    after.push(c.beginTurn("<task>после</task>"));
    const first = blocks(after[0].messages[0]);
    expect(after[0].messages).toHaveLength(1);
    expect(first[0]).toEqual({ type: "text", text: CONTEXT, cache_control: { type: "ephemeral" } });
    const seed = first[1].text!;
    expect(first[1].cache_control).toEqual({ type: "ephemeral" });
    expect(seed).toContain("<earlier_in_meeting>");
    expect(seed).toContain("<summary_so_far>\n**Главное:** обсуждаем релиз.\n</summary_so_far>");
    expect(seed).toContain(`реплика номер ${n} про сроки`);
    expect(seed).not.toContain("реплика номер 1 про сроки");
    expect(seed).toContain("Имена участников: Участник 1 — Айдос.");
    expect(seed).toContain("Айдос: реплика");
    // Nothing is sent twice: the carried lines are not repeated as an update.
    expect(first.slice(2).map((b) => b.text)).toEqual(["<task>после</task>"]);
    expect(breakpointCount(after[0])).toBeLessThanOrEqual(4);
    // Same frozen system prompt and context block: that part of the cache still hits.
    expect(after[0].system).toEqual(before[0].system);
    expect(isPrefix(before.at(-1)!, after[0])).toBe(false);

    c.commit("ок");
    for (let turn = 0; turn < 3; turn++) {
      c.addFinal(seg(++n));
      if (turn === 1) speakers.setName("sys:2", "Дина");
      after.push(c.beginTurn(`<task>после ${turn}</task>`));
      c.commit(`ответ после ${turn}`);
    }
    for (let i = 1; i < after.length; i++) {
      expect(isPrefix(after[i - 1], after[i])).toBe(true);
      expect(breakpointCount(after[i])).toBeLessThanOrEqual(4);
    }
    // A rename after the rollover is announced relative to the names the seed used.
    expect(requestText(after[2] as unknown as CapturedRequest)).toContain("Участник 2 теперь зовут Дина.");
  });

  it("without a summary says so; the fork and the next live turn both start from the seed", () => {
    const c = new Conversation(SYSTEM_PROMPT, CONTEXT, new SpeakerDirectory("me"));
    for (let i = 1; i <= 30; i++) c.addFinal(seg(i));
    c.beginTurn("t");
    c.commit("r");
    const info = c.rollover({ summary: null, maxTranscriptChars: 200 });
    expect(info.summary).toBe(false);
    expect(info.droppedLines).toBeGreaterThan(0);
    const fork = c.fork(rollingSummaryTask({ previous: null, speakers: c.speakers }).text);
    const seed = blocks(fork.messages[0])[1].text!;
    expect(seed).toContain("Резюме ранней части нет");
    expect(seed).toContain("реплика номер 30");
    const live = c.beginTurn("t2");
    expect(isPrefix(fork, live)).toBe(false); // the fork's one-off tail differs…
    expect(blocks(live.messages[0]).slice(0, 2)).toEqual(blocks(fork.messages[0]).slice(0, 2)); // …the seed doesn't
  });

  it("refuses to roll over while a turn awaits its reply", () => {
    const c = new Conversation(SYSTEM_PROMPT, CONTEXT, new SpeakerDirectory("me"));
    c.beginTurn("t");
    expect(() => c.rollover({ summary: null, maxTranscriptChars: 100 })).toThrow();
  });

  it("estimates the next turn from the calibrated tokens-per-character ratio", () => {
    const c = new Conversation(SYSTEM_PROMPT, CONTEXT, new SpeakerDirectory("me"));
    for (let i = 1; i <= 10; i++) c.addFinal(seg(i));
    const est = c.estimateTurn("t");
    const req = c.beginTurn("t");
    expect(est).toBe(Math.ceil(payloadChars(req) * 0.5));
    c.observe(req, 1_000);
    c.commit("r");
    expect(c.tokensPerChar()).toBeCloseTo(1_000 / payloadChars(req));
    expect(c.estimateTurn("t")).toBeGreaterThan(1_000);
  });

  it("the final summary falls back to summary + newest lines when the full transcript would not fit", () => {
    const c = new Conversation(SYSTEM_PROMPT, CONTEXT, new SpeakerDirectory("me"));
    for (let i = 1; i <= 200; i++) c.addFinal(seg(i));
    const task = finalSummaryTask({ speakers: c.speakers }).text;
    const full = c.finalRequest(task);
    expect(requestText(full as unknown as CapturedRequest)).toContain("<full_transcript>");
    const budget = SYSTEM_PROMPT.length + CONTEXT.length + task.length + 4_000;
    const cut = c.finalRequest(task, { maxChars: budget, summary: "Резюме всего." });
    const text = requestText(cut as unknown as CapturedRequest);
    expect(text).not.toContain("<full_transcript>");
    expect(text).toContain("<summary_so_far>\nРезюме всего.\n</summary_so_far>");
    expect(text).toContain("реплика номер 200");
    expect(payloadChars(cut)).toBeLessThanOrEqual(budget);
    // The breakpoint still sits after the transcript part.
    expect(blocks(cut.messages[0])[1].cache_control).toEqual({ type: "ephemeral" });
  });
});

describe("Copilot: rollover before the context window fills up", () => {
  it("tracks usage, asks for a summary near the limit, rolls over past it; append-only holds within each log", async () => {
    const window = contextWindow("claude-haiku-4-5");
    let reported = 2_000;
    // The fake API reports `reported` prompt tokens for the old log; a new log is measured honestly.
    const claude = fakeClaude(
      () => "ок",
      (req) => {
        const prompt = requestText(req).includes("<earlier_in_meeting>") ? Math.ceil(payloadChars(req) * 0.5) : reported;
        return { input_tokens: 50, cache_read_input_tokens: prompt - 150, cache_creation_input_tokens: 100, output_tokens: 20 };
      },
    );
    const usage: string[] = [];
    let pressure = 0;
    const rollovers: RolloverInfo[] = [];
    const speakers = new SpeakerDirectory("me");
    const conv = new Conversation(SYSTEM_PROMPT, CONTEXT, speakers);
    const copilot = new Copilot(conv, {
      getClient: () => claude.client,
      getConfig: () => ({ model: "claude-haiku-4-5", hintEffort: "low", summaryEffort: "high" }),
      getSummary: () => "Резюме до сих пор.",
      onUsage: (out, route) => usage.push(`${route}:${out.usage.cacheRead}`),
      onContextPressure: () => pressure++,
      onRollover: (i) => rollovers.push(i),
    });
    let n = 0;
    const turn = async () => {
      for (let i = 0; i < 3; i++) conv.addFinal(seg(++n));
      await copilot.runLive(hintTask({ lang: "auto", speakers }));
    };

    await turn();
    await turn();
    expect(pressure).toBe(0);
    expect(rollovers).toHaveLength(0);

    // The API now reports a prompt just under the pressure point: the estimate for the next
    // turn (prompt + max_tokens) crosses it, so a fresh summary is requested, no rollover yet.
    reported = Math.floor(window * PREPARE_FRACTION);
    await turn();
    await turn();
    expect(pressure).toBeGreaterThan(0);
    expect(rollovers).toHaveLength(0);

    // Past the rollover point: the next live turn starts a new log.
    reported = Math.floor(window * ROLLOVER_FRACTION);
    await turn();
    await turn();
    expect(rollovers).toHaveLength(1);
    expect(rollovers[0]).toMatchObject({ epoch: 1, summary: true });
    reported = 3_000;
    await turn();
    await turn();

    const calls = claude.calls;
    expect(usage).toHaveLength(calls.length);
    expect(usage.every((u) => u.startsWith("live:"))).toBe(true);
    const cut = calls.findIndex((r) => requestText(r).includes("<earlier_in_meeting>"));
    expect(cut).toBe(5);
    expect(requestText(calls[cut])).toContain("<summary_so_far>\nРезюме до сих пор.\n</summary_so_far>");
    for (let i = 1; i < calls.length; i++) {
      if (i === cut) expect(isPrefix(calls[i - 1], calls[i])).toBe(false);
      else expect(isPrefix(calls[i - 1], calls[i])).toBe(true);
    }
    expect(conv.epoch).toBe(1);
  });

  it("one rollover when the pressure hook starts a rolling-summary fork that rolls over first (as the controller wires it)", async () => {
    const claude = fakeClaude(
      () => "ок",
      () => ({ input_tokens: 150_000 }),
    );
    const speakers = new SpeakerDirectory("me");
    const conv = new Conversation(SYSTEM_PROMPT, CONTEXT, speakers);
    const rollovers: RolloverInfo[] = [];
    const forks: Promise<unknown>[] = [];
    let summarizing = false;
    const copilot: Copilot = new Copilot(conv, {
      getClient: () => claude.client,
      getConfig: () => ({ model: "claude-haiku-4-5", hintEffort: "low", summaryEffort: "high" }),
      getSummary: () => null,
      // SessionController.onContextPressure → generateRolling → runFork, synchronously; the
      // controller marks the summary as streaming first, so the fork's own check doesn't re-enter.
      onContextPressure: () => {
        if (summarizing) return;
        summarizing = true;
        forks.push(copilot.runFork(rollingSummaryTask({ previous: null, speakers })));
      },
      onRollover: (i) => rollovers.push(i),
    });
    for (let i = 1; i <= 20; i++) conv.addFinal(seg(i));
    await copilot.runLive(hintTask({ lang: "auto", speakers })); // calibrates: this text is "huge"
    conv.addFinal(seg(21));
    await copilot.runLive(hintTask({ lang: "auto", speakers }));
    await Promise.all(forks);
    expect(forks).toHaveLength(1);
    expect(rollovers).toHaveLength(1);
    expect(conv.epoch).toBe(1);
  });

  it("rolls over before a fork (rolling summary) that would not fit, and bounds the final summary", async () => {
    // Every response says the text is huge for its size (calibrates a high tokens-per-char).
    const claude = fakeClaude(
      () => "ок",
      () => ({ input_tokens: 150_000 }),
    );
    const speakers = new SpeakerDirectory("me");
    const conv = new Conversation(SYSTEM_PROMPT, CONTEXT, speakers);
    const rollovers: RolloverInfo[] = [];
    const copilot = new Copilot(conv, {
      getClient: () => claude.client,
      getConfig: () => ({ model: "claude-haiku-4-5", hintEffort: "low", summaryEffort: "high" }),
      getSummary: () => null,
      onRollover: (i) => rollovers.push(i),
    });
    for (let i = 1; i <= 20; i++) conv.addFinal(seg(i));
    await copilot.runLive(hintTask({ lang: "auto", speakers })); // calibrates: this text is "huge"
    conv.addFinal(seg(21));
    await copilot.runFork(rollingSummaryTask({ previous: null, speakers }));
    expect(rollovers).toHaveLength(1);
    expect(requestText(claude.calls[1])).toContain("<earlier_in_meeting>");
    await copilot.runFinal(finalSummaryTask({ speakers }));
    const final = requestText(claude.calls[2]);
    expect(final).not.toContain("<full_transcript>");
    expect(final).toContain("реплика номер 21");
  });
});
