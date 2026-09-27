import { describe, expect, it } from "vitest";
import type { Segment } from "../types";
import {
  applyRelabel,
  compareLabels,
  defaultSpeakerName,
  effectiveLabel,
  formatTalkTime,
  inferMicMode,
  isRenamable,
  PALETTE_SIZE,
  pluralTurns,
  speakerClass,
  SpeakerDirectory,
  speakerStats,
} from "./speakers";

function seg(id: string, speaker: string | null, startMs: number, endMs: number, source: Segment["source"] = "system"): Segment {
  return { id, source, speaker, startMs, endMs, text: "текст", isFinal: true };
}

describe("default names and labels", () => {
  it("maps labels to display names", () => {
    expect(defaultSpeakerName("me")).toBe("Я");
    expect(defaultSpeakerName("sys:3")).toBe("Участник 3");
    expect(defaultSpeakerName("mic:12")).toBe("Зал 12");
    expect(defaultSpeakerName(null)).toBe("?");
    expect(defaultSpeakerName(undefined)).toBe("?");
    expect(defaultSpeakerName("guest")).toBe("guest");
  });

  it("in micMode \"me\" an unlabelled mic segment (a partial, an old backend) is the user", () => {
    expect(effectiveLabel({ source: "mic", speaker: null }, "me")).toBe("me");
    expect(effectiveLabel({ source: "mic", speaker: null }, "room")).toBeNull();
    expect(effectiveLabel({ source: "system", speaker: null }, "me")).toBeNull();
    expect(effectiveLabel({ source: "mic", speaker: "mic:2" }, "room")).toBe("mic:2");
  });

  it("only other people's labels are renamable", () => {
    expect(isRenamable("sys:1")).toBe(true);
    expect(isRenamable("mic:4")).toBe(true);
    expect(isRenamable("me")).toBe(false);
    expect(isRenamable(null)).toBe(false);
  });

  it("orders labels: me, sys by number, mic by number, others", () => {
    expect(["mic:2", "sys:10", "x", "me", "sys:2", null, "mic:1"].sort(compareLabels)).toEqual(["me", "sys:2", "sys:10", "mic:1", "mic:2", "x", null]);
  });
});

describe("speaker colors", () => {
  it("are stable per label and stay within the palette", () => {
    expect(speakerClass("sys:1")).toBe("spk-1");
    expect(speakerClass("sys:1")).toBe(speakerClass("sys:1"));
    expect(speakerClass(`sys:${PALETTE_SIZE + 1}`)).toBe("spk-1");
    expect(speakerClass("me")).toBe("spk-me");
    expect(speakerClass(null)).toBe("spk-none");
    for (let n = 1; n <= 40; n++) {
      for (const p of ["sys", "mic"]) {
        const i = Number(speakerClass(`${p}:${n}`).slice(4));
        expect(i).toBeGreaterThanOrEqual(1);
        expect(i).toBeLessThanOrEqual(PALETTE_SIZE);
      }
    }
    expect(speakerClass("custom-label")).toMatch(/^spk-\d+$/);
  });

  it("start the room's voices half a palette away from the call's", () => {
    expect(speakerClass("mic:1")).not.toBe(speakerClass("sys:1"));
    expect(speakerClass("mic:1")).toBe(`spk-${PALETTE_SIZE / 2 + 1}`);
  });
});

describe("SpeakerDirectory", () => {
  it("resolves custom names over defaults; the default name or empty resets", () => {
    const d = new SpeakerDirectory("room", { "sys:3": " Айдос  ", me: "Хуго" });
    expect(d.nameOf("sys:3")).toBe("Айдос");
    expect(d.nameOf("me")).toBe("Я"); // «Я» can't be renamed
    expect(d.nameFor({ source: "mic", speaker: null })).toBe("?");
    expect(d.setName("sys:3", "Айдос")).toBe(false);
    expect(d.setName("sys:3", "Участник 3")).toBe(true);
    expect(d.customName("sys:3")).toBeNull();
    d.setName("mic:2", "Мария");
    d.setName("sys:1", "Айгерим");
    expect(d.setName("sys:1", "")).toBe(true);
    d.setName("sys:5", "  Ерлан   Н.  ");
    expect(d.snapshot()).toEqual({ "sys:5": "Ерлан Н.", "mic:2": "Мария" });
    expect(Object.keys(d.snapshot())).toEqual(["sys:5", "mic:2"]);
  });
});

describe("speakerStats", () => {
  const d = new SpeakerDirectory("room", { "sys:2": "Айдос" });
  const finals = [
    seg("1", "sys:1", 0, 4_000),
    seg("2", "sys:1", 4_000, 6_000), // same turn, split by the recognizer
    seg("3", "sys:2", 7_000, 17_000),
    seg("4", "sys:1", 18_000, 19_000),
    seg("5", null, 20_000, 20_500, "mic"),
    seg("6", "me", 21_000, 23_000, "mic"),
    { ...seg("7", "sys:3", 24_000, 30_000), text: "  " }, // empty: ignored
  ];

  it("sums talk time and counts turns as runs, sorted by talk time, unknown last", () => {
    const stats = speakerStats(finals, d);
    expect(stats.map((s) => [s.label, s.name, s.talkMs, s.turns, s.segments, s.custom])).toEqual([
      ["sys:2", "Айдос", 10_000, 1, 1, true],
      ["sys:1", "Участник 1", 7_000, 2, 3, false],
      ["me", "Я", 2_000, 1, 1, false],
      [null, "?", 500, 1, 1, false],
    ]);
  });

  it("adds named speakers with no segments; ties sort by label", () => {
    const stats = speakerStats([seg("1", "sys:4", 0, 1_000), seg("2", "sys:1", 2_000, 3_000)], d, ["sys:2"]);
    expect(stats.map((s) => [s.label, s.talkMs])).toEqual([
      ["sys:1", 1_000],
      ["sys:4", 1_000],
      ["sys:2", 0],
    ]);
  });
});

describe("relabel and helpers", () => {
  it("applyRelabel updates only changed segments and keeps identity when nothing changes", () => {
    const list = [seg("a", "sys:5", 0, 1), seg("b", null, 1, 2), seg("c", "sys:1", 2, 3)];
    const out = applyRelabel(list, [
      { segmentId: "a", speaker: "sys:2" },
      { segmentId: "b", speaker: "sys:2" },
      { segmentId: "c", speaker: "sys:1" },
      { segmentId: "zzz", speaker: "sys:9" },
    ]);
    expect(out.map((s) => s.speaker)).toEqual(["sys:2", "sys:2", "sys:1"]);
    expect(out[2]).toBe(list[2]);
    expect(applyRelabel(list, [{ segmentId: "c", speaker: "sys:1" }])).toBe(list);
    expect(applyRelabel(list, [])).toBe(list);
  });

  it("infers the mic mode of a stored meeting", () => {
    expect(inferMicMode([{ speaker: "sys:1" }, { speaker: "me" }])).toBe("me");
    expect(inferMicMode([{ speaker: "mic:1" }])).toBe("room");
  });

  it("formats talk time and turn counts in Russian", () => {
    expect(formatTalkTime(45_400)).toBe("45 с");
    expect(formatTalkTime(185_000)).toBe("3 мин 05 с");
    expect(formatTalkTime(3_725_000)).toBe("1 ч 02 мин");
    expect([1, 2, 5, 11, 21, 22, 112].map(pluralTurns)).toEqual([
      "1 реплика",
      "2 реплики",
      "5 реплик",
      "11 реплик",
      "21 реплика",
      "22 реплики",
      "112 реплик",
    ]);
  });
});
