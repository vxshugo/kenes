import { describe, expect, it } from "vitest";
import type { Segment } from "../types";
import { Conversation, type MessageParam, type RequestPayload } from "./conversation";
import { SYSTEM_PROMPT, formatContextBlock } from "./prompts";
import { SpeakerDirectory } from "./speakers";
import { finalSummaryTask, hintTask, rollingSummaryTask } from "./tasks";

const CONTEXT = formatContextBlock({
  title: "Синк",
  context: "повестка: релиз",
  profile: "техлид",
  setup: { captureMic: true, captureSystem: true, micMode: "me" },
  voiceprint: false,
  myNames: ["Хуго"],
  date: "2026-09-27, воскресенье",
});

/** mic → "me", system → "sys:1" unless a label is given. */
function seg(id: string, source: Segment["source"], startMs: number, text: string, speaker?: string | null): Segment {
  const label = speaker !== undefined ? speaker : source === "mic" ? "me" : "sys:1";
  return { id, source, speaker: label, startMs, endMs: startMs + 1500, text, isFinal: true };
}

function newConversation(speakers = new SpeakerDirectory("me")) {
  return new Conversation(SYSTEM_PROMPT, CONTEXT, speakers);
}

type Block = { type: string; text?: string; cache_control?: unknown };

function blocks(m: MessageParam): Block[] {
  return (typeof m.content === "string" ? [{ type: "text", text: m.content }] : m.content) as Block[];
}

/** All cache_control locations as [messageIndex, blockIndex]; system is messageIndex -1. */
function breakpoints(p: RequestPayload): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  p.system.forEach((b, i) => b.cache_control && out.push([-1, i]));
  p.messages.forEach((m, mi) => blocks(m).forEach((b, bi) => b.cache_control && out.push([mi, bi])));
  return out;
}

/** The request with every cache_control marker removed (markers move; bytes must not). */
function strip(p: RequestPayload): unknown {
  const clean = (b: Block) => {
    const { cache_control: _cc, ...rest } = b;
    return rest;
  };
  return {
    system: p.system.map((b) => clean(b as Block)),
    messages: p.messages.map((m) => ({ role: m.role, content: blocks(m).map(clean) })),
  };
}

function isPrefix(prev: RequestPayload, next: RequestPayload): boolean {
  const a = strip(prev) as { system: unknown[]; messages: unknown[] };
  const b = strip(next) as { system: unknown[]; messages: unknown[] };
  if (JSON.stringify(a.system) !== JSON.stringify(b.system)) return false;
  if (a.messages.length > b.messages.length) return false;
  return a.messages.every((m, i) => JSON.stringify(m) === JSON.stringify(b.messages[i]));
}

describe("Conversation: layout and cache breakpoints", () => {
  it("first turn: frozen system ◆, context block ◆ first, task ◆ last", () => {
    const c = newConversation();
    c.addFinal(seg("system-1", "system", 3_000, "когда   релиз"));
    const p = c.beginTurn("<task>hint</task>");

    expect(p.system).toEqual([{ type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } }]);
    expect(p.messages).toHaveLength(1);
    const content = blocks(p.messages[0]);
    expect(p.messages[0].role).toBe("user");
    expect(content[0]).toEqual({ type: "text", text: CONTEXT, cache_control: { type: "ephemeral" } });
    expect(content[1]).toEqual({
      type: "text",
      text: "<transcript_update>\n[00:03] Участник 1: когда релиз\n</transcript_update>",
    });
    expect(content[2]).toEqual({ type: "text", text: "<task>hint</task>", cache_control: { type: "ephemeral" } });
    expect(breakpoints(p)).toEqual([
      [-1, 0],
      [0, 0],
      [0, 2],
    ]);
  });

  it("later turns: the moving breakpoint is only on the newest user block (max 3 in total)", () => {
    const c = newConversation();
    c.addFinal(seg("system-1", "system", 3_000, "когда релиз"));
    c.beginTurn("<task>1</task>");
    c.commit("в пятницу");
    c.addFinal(seg("mic-1", "mic", 9_000, "в пятницу"));
    const p = c.beginTurn("<task>2</task>");

    expect(p.messages.map((m) => m.role)).toEqual(["user", "assistant", "user"]);
    expect(blocks(p.messages[1])).toEqual([{ type: "text", text: "в пятницу" }]);
    expect(blocks(p.messages[2])).toEqual([
      { type: "text", text: "<transcript_update>\n[00:09] Я: в пятницу\n</transcript_update>" },
      { type: "text", text: "<task>2</task>", cache_control: { type: "ephemeral" } },
    ]);
    expect(breakpoints(p)).toEqual([
      [-1, 0],
      [0, 0],
      [2, 1],
    ]);
  });

  it("omits the transcript block when nothing new was said", () => {
    const c = newConversation();
    const p = c.beginTurn("<task>x</task>");
    expect(blocks(p.messages[0]).map((b) => b.text)).toEqual([CONTEXT, "<task>x</task>"]);
  });

  it("orders a transcript update by start time, then id", () => {
    const c = newConversation();
    c.addFinal(seg("system-2", "system", 5_000, "б"));
    c.addFinal(seg("mic-1", "mic", 2_000, "а"));
    c.addFinal(seg("system-1", "system", 5_000, "в"));
    const p = c.beginTurn("t");
    expect(blocks(p.messages[0])[1].text).toBe(
      "<transcript_update>\n[00:02] Я: а\n[00:05] Участник 1: в\n[00:05] Участник 1: б\n</transcript_update>",
    );
  });

  it("ignores partials, duplicates and empty finals", () => {
    const c = newConversation();
    expect(c.addFinal({ ...seg("s-1", "system", 0, "частично"), isFinal: false })).toBe(false);
    expect(c.addFinal(seg("s-2", "system", 0, "   "))).toBe(false);
    expect(c.addFinal(seg("s-3", "system", 0, "да"))).toBe(true);
    expect(c.addFinal(seg("s-3", "system", 0, "да"))).toBe(false);
    expect(c.pendingCount()).toBe(1);
  });
});

describe("Conversation: append-only property", () => {
  it("every request is a byte-identical prefix of the next one (markers aside)", () => {
    const c = newConversation();
    const requests: RequestPayload[] = [];
    let t = 0;
    for (let i = 0; i < 6; i++) {
      c.addFinal(seg(`system-${i}`, "system", (t += 4_000), `вопрос номер ${i} когда`));
      c.addFinal(seg(`mic-${i}`, "mic", (t += 3_000), `ответ ${i}`));
      requests.push(c.beginTurn(hintTask({ lang: "auto", speakers: c.speakers }).text));
      c.commit(`подсказка ${i}`);
    }
    for (let i = 1; i < requests.length; i++) {
      expect(isPrefix(requests[i - 1], requests[i])).toBe(true);
      expect(breakpoints(requests[i]).length).toBeLessThanOrEqual(4);
    }
  });

  it("rollback drops only the failed trailing user turn and requeues its lines", () => {
    const c = newConversation();
    c.addFinal(seg("system-1", "system", 1_000, "первый"));
    const ok = c.beginTurn("t1");
    c.commit("r1");
    c.addFinal(seg("system-2", "system", 2_000, "второй"));
    c.beginTurn("t2");
    expect(c.awaitingReply).toBe(true);
    c.rollback();
    expect(c.awaitingReply).toBe(false);
    expect(c.exchanges).toBe(1);
    c.addFinal(seg("system-3", "system", 3_000, "третий"));
    const retry = c.beginTurn("t3");
    expect(isPrefix(ok, retry)).toBe(true);
    // The failed turn's line rides along with the next turn.
    expect(blocks(retry.messages[2])[0].text).toBe(
      "<transcript_update>\n[00:02] Участник 1: второй\n[00:03] Участник 1: третий\n</transcript_update>",
    );
  });

  it("rollback is a no-op after a commit", () => {
    const c = newConversation();
    c.beginTurn("t1");
    c.commit("r1");
    c.rollback();
    expect(c.exchanges).toBe(1);
  });

  it("refuses overlapping turns and empty replies", () => {
    const c = newConversation();
    c.beginTurn("t1");
    expect(() => c.beginTurn("t2")).toThrow();
    expect(() => c.commit("   ")).toThrow();
  });
});

describe("Conversation: fork (rolling summary)", () => {
  it("reuses the committed prefix, puts the breakpoint on the last shared block, leaves the log alone", () => {
    const c = newConversation();
    c.addFinal(seg("system-1", "system", 1_000, "первый"));
    const live1 = c.beginTurn("t1");
    c.commit("r1");
    c.addFinal(seg("system-2", "system", 2_000, "второй"));

    const fork = c.fork(rollingSummaryTask({ previous: null, speakers: c.speakers }).text);
    expect(isPrefix(live1, fork)).toBe(true);
    expect(breakpoints(fork)).toEqual([
      [-1, 0],
      [0, 0],
      [1, 0], // the last committed (assistant) block, not the one-off task
    ]);
    expect(blocks(fork.messages[2])[0].text).toContain("второй");

    // The log is untouched and the line is still pending for the next live turn.
    expect(c.exchanges).toBe(1);
    expect(c.pendingCount()).toBe(1);
    const live2 = c.beginTurn("t2");
    expect(blocks(live2.messages[2])[0].text).toContain("второй");
  });

  it("while a live turn is in flight, forks from the committed part and includes its lines", () => {
    const c = newConversation();
    c.beginTurn("t1");
    c.commit("r1");
    c.addFinal(seg("system-2", "system", 2_000, "в полёте"));
    c.beginTurn("t2");
    c.addFinal(seg("system-3", "system", 3_000, "новое"));
    const fork = c.fork("summary");
    expect(fork.messages.map((m) => m.role)).toEqual(["user", "assistant", "user"]);
    expect(blocks(fork.messages[2])[0].text).toBe(
      "<transcript_update>\n[00:02] Участник 1: в полёте\n[00:03] Участник 1: новое\n</transcript_update>",
    );
    expect(c.awaitingReply).toBe(true);
  });

  it("with an empty log, only system and context carry breakpoints", () => {
    const c = newConversation();
    c.addFinal(seg("system-1", "system", 1_000, "первый"));
    expect(breakpoints(c.fork("summary"))).toEqual([
      [-1, 0],
      [0, 0],
    ]);
  });
});

describe("Conversation: final summary request", () => {
  it("sends the full transcript once with a breakpoint after it", () => {
    const c = newConversation();
    c.addFinal(seg("mic-1", "mic", 61_000, "мы успеем"));
    c.addFinal(seg("system-1", "system", 5_000, "успеете к пятнице"));
    c.beginTurn("t1");
    c.commit("r1");
    const task = finalSummaryTask({ speakers: c.speakers }).text;
    const p = c.finalRequest(task);
    expect(p.system[0].text).toBe(SYSTEM_PROMPT);
    expect(p.messages).toHaveLength(1);
    const content = blocks(p.messages[0]);
    expect(content.map((b) => b.text)).toEqual([
      CONTEXT,
      "<full_transcript>\n[00:05] Участник 1: успеете к пятнице\n[01:01] Я: мы успеем\n</full_transcript>",
      task,
    ]);
    expect(breakpoints(p)).toEqual([
      [-1, 0],
      [0, 1],
    ]);
  });
});

describe("Conversation: speakers", () => {
  it("renders display names: «Я», «Участник N», «Зал N», custom names and «?»", () => {
    const speakers = new SpeakerDirectory("room", { "sys:3": "Айдос" });
    const c = newConversation(speakers);
    c.addFinal(seg("system-1", "system", 1_000, "айдос что скажешь", "sys:1"));
    c.addFinal(seg("system-2", "system", 2_000, "думаю успеем", "sys:3"));
    c.addFinal(seg("mic-1", "mic", 3_000, "а в зале согласны", "mic:2"));
    c.addFinal(seg("mic-2", "mic", 4_000, "да", null));
    c.addFinal(seg("mic-3", "mic", 5_000, "я тоже за", "me"));
    const p = c.beginTurn("t");
    expect(blocks(p.messages[0])[1].text).toBe(
      "<transcript_update>\n[00:01] Участник 1: айдос что скажешь\n[00:02] Айдос: думаю успеем\n[00:03] Зал 2: а в зале согласны\n[00:04] ?: да\n[00:05] Я: я тоже за\n</transcript_update>",
    );
  });

  it("a rename never rewrites earlier turns: it adds a note to the next user turn (prefix property holds)", () => {
    const speakers = new SpeakerDirectory("me");
    const c = newConversation(speakers);
    c.addFinal(seg("system-1", "system", 1_000, "айдос что по мобилке", "sys:1"));
    c.addFinal(seg("system-2", "system", 2_000, "выкатили на десять процентов", "sys:3"));
    const r1 = c.beginTurn("t1");
    c.commit("r1");

    speakers.setName("sys:3", "Айдос");
    c.addFinal(seg("system-3", "system", 9_000, "крэшей нет", "sys:3"));
    const r2 = c.beginTurn("t2");
    expect(isPrefix(r1, r2)).toBe(true);
    // The old line still says «Участник 3»…
    expect(blocks(r2.messages[0])[1].text).toContain("[00:02] Участник 3: выкатили на десять процентов");
    // …and the new turn starts with the note, then lines with the new name, then the task.
    expect(blocks(r2.messages[2]).map((b) => b.text)).toEqual([
      "<speaker_names>\nУчастник 3 теперь зовут Айдос.\n</speaker_names>",
      "<transcript_update>\n[00:09] Айдос: крэшей нет\n</transcript_update>",
      "t2",
    ]);
    c.commit("r2");

    // No repeated note; a second rename and a reset are notes too, in label order.
    speakers.setName("sys:3", "Айдос Сейтказиев");
    speakers.setName("sys:1", "Айгерим");
    const r3 = c.beginTurn("t3");
    expect(isPrefix(r2, r3)).toBe(true);
    expect(blocks(r3.messages[4])[0].text).toBe(
      "<speaker_names>\nУчастник 1 теперь зовут Айгерим.\nАйдос теперь зовут Айдос Сейтказиев.\n</speaker_names>",
    );
    c.commit("r3");
    speakers.setName("sys:1", "");
    const r4 = c.beginTurn("t4");
    expect(isPrefix(r3, r4)).toBe(true);
    expect(blocks(r4.messages[6])[0].text).toBe("<speaker_names>\nАйгерим — снова Участник 1 (имя снято).\n</speaker_names>");
    expect(breakpoints(r4).length).toBeLessThanOrEqual(4);
  });

  it("no note for a label whose lines were never sent; renaming back before the next turn is a no-op", () => {
    const speakers = new SpeakerDirectory("me");
    const c = newConversation(speakers);
    c.addFinal(seg("system-1", "system", 1_000, "привет", "sys:1"));
    c.beginTurn("t1");
    c.commit("r1");
    speakers.setName("sys:2", "Дина"); // never spoke in the log yet
    speakers.setName("sys:1", "Ерлан");
    speakers.setName("sys:1", ""); // back to the default before anything was sent
    c.addFinal(seg("system-2", "system", 2_000, "регресс в среду", "sys:2"));
    const p = c.beginTurn("t2");
    expect(blocks(p.messages[2]).map((b) => b.text)).toEqual([
      "<transcript_update>\n[00:02] Дина: регресс в среду\n</transcript_update>",
      "t2",
    ]);
  });

  it("rollback returns the rename note to the queue; fork includes it without consuming it", () => {
    const speakers = new SpeakerDirectory("me");
    const c = newConversation(speakers);
    c.addFinal(seg("system-1", "system", 1_000, "вопрос", "sys:2"));
    c.beginTurn("t1");
    c.commit("r1");
    speakers.setName("sys:2", "Дина");
    c.beginTurn("t2");
    c.rollback();

    const fork = c.fork("summary");
    expect(blocks(fork.messages[2])[0].text).toBe("<speaker_names>\nУчастник 2 теперь зовут Дина.\n</speaker_names>");
    const retry = c.beginTurn("t3");
    expect(blocks(retry.messages[2])[0].text).toBe("<speaker_names>\nУчастник 2 теперь зовут Дина.\n</speaker_names>");
  });

  it("while a live turn is in flight, a fork carries notes relative to the committed log", () => {
    const speakers = new SpeakerDirectory("me");
    const c = newConversation(speakers);
    c.addFinal(seg("system-1", "system", 1_000, "вопрос", "sys:2"));
    c.beginTurn("t1");
    c.commit("r1");
    speakers.setName("sys:2", "Дина");
    c.addFinal(seg("system-2", "system", 2_000, "ещё", "sys:2"));
    c.beginTurn("t2"); // in flight: carries the note
    const fork = c.fork("summary");
    expect(blocks(fork.messages[2]).map((b) => b.text)).toEqual([
      "<speaker_names>\nУчастник 2 теперь зовут Дина.\n</speaker_names>",
      "<transcript_update>\n[00:02] Дина: ещё\n</transcript_update>",
      "summary",
    ]);
  });

  it("relabel: unsent lines and the final summary use the new label; sent lines are not rewritten", () => {
    const speakers = new SpeakerDirectory("me", { "sys:2": "Ерлан" });
    const c = newConversation(speakers);
    c.addFinal(seg("system-1", "system", 1_000, "первый", "sys:5"));
    const r1 = c.beginTurn("t1");
    c.commit("r1");
    c.addFinal(seg("system-2", "system", 2_000, "да", null));
    expect(c.relabel([{ segmentId: "system-1", speaker: "sys:2" }, { segmentId: "system-2", speaker: "sys:2" }, { segmentId: "nope", speaker: "sys:1" }])).toBe(2);
    const r2 = c.beginTurn("t2");
    expect(isPrefix(r1, r2)).toBe(true);
    expect(blocks(r2.messages[2])[0].text).toBe("<transcript_update>\n[00:02] Ерлан: да\n</transcript_update>");
    const final = c.finalRequest("final");
    expect(blocks(final.messages[0])[1].text).toBe("<full_transcript>\n[00:01] Ерлан: первый\n[00:02] Ерлан: да\n</full_transcript>");
  });
});
