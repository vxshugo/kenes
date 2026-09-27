import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EMPTY_USAGE } from "../llm/usage";
import { MemoryStorage } from "../test/helpers";
import type { Note, Segment } from "../types";
import { cardMeta, cardsFromNotes, inferMicMode, latestNote, loadUsage, phaseFor, saveUsage } from "./restore";

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
});
afterEach(() => {
  vi.unstubAllGlobals();
});

const note = (id: string, kind: Note["kind"], trigger: string | null, createdAt: string, content = id): Note => ({
  id,
  meetingId: "m",
  kind,
  content,
  trigger,
  createdAt,
});

describe("restoring a running meeting", () => {
  it("maps saved note triggers back to card kinds and labels", () => {
    expect(cardMeta("manual")).toEqual({ kind: "hint", label: "Что ответить?", detail: null, auto: false });
    expect(cardMeta("auto: когда релиз")).toEqual({ kind: "hint", label: "Подсказка (авто)", detail: "когда релиз", auto: true });
    expect(cardMeta("ask: что с бюджетом")).toMatchObject({ kind: "ask", detail: "что с бюджетом" });
    expect(cardMeta("explain: SLA")).toMatchObject({ kind: "explain", label: "Объяснение", detail: "SLA" });
    expect(cardMeta("translate")).toMatchObject({ kind: "translate", label: "Перевод" });
    expect(cardMeta("recap: 5m")).toMatchObject({ kind: "recap", label: "Кратко: 5 мин" });
    expect(cardMeta(null)).toMatchObject({ kind: "hint", label: "Подсказка", detail: null });
  });

  it("turns hint notes into finished cards, newest first, capped", () => {
    const notes = [
      note("a", "hint", "manual", "2026-09-27T10:00:00Z"),
      note("s", "summary", "rolling", "2026-09-27T10:01:00Z"),
      note("b", "hint", "recap: 5m", "2026-09-27T10:02:00Z"),
      note("c", "hint", "auto: вопрос", "2026-09-27T10:03:00Z"),
    ];
    const cards = cardsFromNotes(notes, 2);
    expect(cards.map((c) => c.id)).toEqual(["restored-c", "restored-b"]);
    expect(cards[0]).toMatchObject({ status: "done", text: "c", hidden: false, auto: true, detail: "вопрос" });
    expect(latestNote(notes, "summary")?.id).toBe("s");
    expect(latestNote(notes, "final")).toBeNull();
  });

  it("maps the pipeline state to a phase and infers the mic mode from labels", () => {
    expect(phaseFor("running")).toBe("running");
    expect(phaseFor("loading")).toBe("loading");
    expect(phaseFor("idle")).toBe("loading");
    expect(phaseFor("error")).toBe("error");
    const s = (speaker: string | null): Segment => ({ id: "x", source: "mic", speaker, startMs: 0, endMs: 1, text: "a", isFinal: true });
    expect(inferMicMode([s("me"), s("sys:1")], "me")).toBe("me");
    expect(inferMicMode([s("mic:2")], "me")).toBe("room");
  });

  it("keeps usage per meeting in localStorage", () => {
    expect(loadUsage("m1")).toBeNull();
    saveUsage("m1", { ...EMPTY_USAGE, requests: 2, cost: 0.5 });
    expect(loadUsage("m1")).toMatchObject({ requests: 2, cost: 0.5 });
    localStorage.setItem("kenes.usage.m2", "{broken");
    expect(loadUsage("m2")).toBeNull();
  });
});
