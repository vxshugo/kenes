import { describe, expect, it } from "vitest";
import type { Segment } from "../types";
import {
  SYSTEM_PROMPT,
  describeSetup,
  formatClock,
  formatContextBlock,
  formatMeetingDate,
  formatSegmentLine,
  formatTranscript,
  hasKazakhLetters,
  languageInstruction,
} from "./prompts";
import { SpeakerDirectory } from "./speakers";
import {
  askTask,
  budgetFor,
  explainTask,
  finalSummaryTask,
  hintTask,
  isSkipReply,
  mayBeSkip,
  namesMapLine,
  pickTranslationLines,
  recapTask,
  rollingSummaryTask,
  translateTask,
} from "./tasks";

function seg(p: Partial<Segment>): Segment {
  return { id: "x", source: "system", speaker: "sys:1", startMs: 0, endMs: 0, text: "", isFinal: true, ...p };
}

const SPEAKERS = new SpeakerDirectory("me");

describe("formatting", () => {
  it("formats the session clock", () => {
    expect(formatClock(0)).toBe("00:00");
    expect(formatClock(59_999)).toBe("00:59");
    expect(formatClock(12 * 60_000 + 3_000)).toBe("12:03");
    expect(formatClock(3_600_000 + 65_000)).toBe("1:01:05");
    expect(formatClock(-5)).toBe("00:00");
  });

  it("renders a transcript line deterministically, with the speaker's display name", () => {
    const line = formatSegmentLine(seg({ startMs: 723_400, text: "  а когда   релиз \n" }), SPEAKERS);
    expect(line).toBe("[12:03] Участник 1: а когда релиз");
    const named = new SpeakerDirectory("me", { "sys:1": "Айдос" });
    expect(formatSegmentLine(seg({ startMs: 723_400, text: "а когда релиз" }), named)).toBe("[12:03] Айдос: а когда релиз");
  });

  it("sorts and skips empty lines in a transcript", () => {
    const t = formatTranscript(
      [
        seg({ id: "b", source: "mic", speaker: "me", startMs: 2_000, text: "второй" }),
        seg({ id: "a", startMs: 1_000, text: "первый" }),
        seg({ id: "c", startMs: 3_000, text: "  " }),
      ],
      SPEAKERS,
    );
    expect(t).toBe("[00:01] Участник 1: первый\n[00:02] Я: второй");
  });

  it("detects Kazakh-only letters", () => {
    expect(hasKazakhLetters("қашан")).toBe(true);
    expect(hasKazakhLetters("БІЗ")).toBe(true);
    expect(hasKazakhLetters("когда релиз")).toBe(false);
  });

  it("formats the meeting date with a Russian weekday", () => {
    expect(formatMeetingDate(new Date(2026, 8, 27))).toBe("2026-09-27, воскресенье");
  });
});

describe("system prompt", () => {
  it("is frozen: no dates, clock times or template holes", () => {
    expect(SYSTEM_PROMPT).not.toMatch(/\$\{|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}/);
  });
  it("covers the input format, speakers, facts and language rules", () => {
    for (const needle of ["строчными", "без знаков препинания", "казах", "«Я»", "выдумывай", "<tone_preference>"]) {
      expect(SYSTEM_PROMPT).toContain(needle);
    }
  });
  it("says the meeting is multi-party and explains labels, rename notes and SKIP", () => {
    for (const needle of ["10–15 человек", "«Участник N»", "«Зал N»", "«?»", "<speaker_names>", "теперь зовут", "SKIP", "Диаризация ошибается"]) {
      expect(SYSTEM_PROMPT).toContain(needle);
    }
  });
  it("is long enough to be cached on its own (Claude Opus 5 minimum is 512 tokens)", () => {
    // Cyrillic runs well under 4 chars/token; 3000 chars is comfortably above 512 tokens.
    expect(SYSTEM_PROMPT.length).toBeGreaterThan(3000);
  });
});

describe("meeting context block", () => {
  const base = {
    title: "Синк  по  релизу",
    context: "повестка",
    profile: "техлид",
    setup: { captureMic: true, captureSystem: true, micMode: "me" as const },
    voiceprint: false,
    myNames: ["Хуго", " Hugo "],
    date: "2026-09-27, воскресенье",
  };

  it("is deterministic for the same input", () => {
    expect(formatContextBlock(base)).toBe(formatContextBlock({ ...base }));
  });

  it("contains title, date, format, the user's names, profile and agenda in fixed order", () => {
    const b = formatContextBlock(base);
    expect(b).toBe(
      [
        "<meeting_context>",
        "<meeting_title>Синк по релизу</meeting_title>",
        "<meeting_date>2026-09-27, воскресенье</meeting_date>",
        "<meeting_format>онлайн-звонок, пользователь в наушниках: «Я» — микрофон пользователя, «Участник N» — голоса остальных из звонка</meeting_format>",
        "<user_names>Хуго, Hugo — так к пользователю («Я») обращаются на встрече</user_names>",
        "<user_profile>",
        "техлид",
        "</user_profile>",
        "<agenda_and_notes>",
        "повестка",
        "</agenda_and_notes>",
        "</meeting_context>",
      ].join("\n"),
    );
  });

  it("marks missing fields and describes room and hybrid formats", () => {
    const b = formatContextBlock({ ...base, profile: " ", context: "", myNames: [], setup: { captureMic: true, captureSystem: false, micMode: "room" } });
    expect(b).toContain("<user_profile>\n(не указано)\n</user_profile>");
    expect(b).toContain("<user_names>(не указано)</user_names>");
    expect(b).toContain("очная встреча в зале");
    expect(b).toContain("образца голоса пользователя нет");
    const hybrid = describeSetup({ captureMic: true, captureSystem: true, micMode: "room" }, true);
    expect(hybrid).toContain("гибридная встреча");
    expect(hybrid).toContain("«Зал N»");
    expect(hybrid).toContain("«Участник N»");
    expect(hybrid).toContain("по образцу его голоса");
  });
});

describe("tasks", () => {
  it("hint: manual vs automatic, with language line and latency hint", () => {
    const manual = hintTask({ lang: "auto", speakers: SPEAKERS });
    expect(manual.trigger).toBe("manual");
    expect(manual.skippable).toBe(false);
    expect(manual.route).toBe("live");
    expect(manual.text).toMatch(/^<task type="hint">\n[\s\S]*\n<\/task>$/);
    expect(manual.text).toContain(languageInstruction("auto"));
    expect(manual.text).toContain("Latency-sensitive");

    const auto = hintTask({ lang: "kk", speakers: SPEAKERS, question: seg({ text: "қашан   бастаймыз" }), auto: "any" });
    expect(auto.trigger).toBe("auto: қашан бастаймыз");
    expect(auto.text).toContain("Участник 1: «қашан бастаймыз»");
    expect(auto.text).toContain(languageInstruction("kk"));
    expect(auto.text).not.toContain("SKIP");
  });

  it("hint includes in-progress partials", () => {
    const t = hintTask({ lang: "ru", speakers: SPEAKERS, partials: [seg({ startMs: 5_000, text: "а сколько", isFinal: false })] });
    expect(t.text).toContain("[00:05] Участник 1: а сколько …");
  });

  it("language instructions differ per setting", () => {
    expect(languageInstruction("auto")).toContain("казахском");
    expect(languageInstruction("ru")).toContain("русский");
    expect(languageInstruction("kk")).toContain("казахский");
  });

  it("ask / explain / recap / translate / rolling / final have the right routes", () => {
    expect(askTask({ question: "кто такой ерлан", lang: "auto", speakers: SPEAKERS }).route).toBe("live");
    expect(explainTask({ term: "sla", lang: "auto" }).text).toContain("«sla»");
    expect(recapTask({ minutes: 5, nowMs: 7 * 60_000, speakers: SPEAKERS }).text).toContain("[02:00]");
    expect(rollingSummaryTask({ previous: "старое", speakers: SPEAKERS }).route).toBe("fork");
    expect(rollingSummaryTask({ previous: "старое", speakers: SPEAKERS }).text).toContain("<previous_summary>\nстарое\n</previous_summary>");
    expect(rollingSummaryTask({ previous: null, speakers: SPEAKERS }).text).not.toContain("previous_summary");
    const final = finalSummaryTask({ speakers: SPEAKERS });
    expect(final.route).toBe("final");
    for (const h of ["## Итоги", "## Решения", "## Задачи", "## Участники и позиции", "## Открытые вопросы", "## Ключевые цифры"]) {
      expect(final.text).toContain(h);
    }
    expect(final.text.indexOf("## Задачи")).toBeLessThan(final.text.indexOf("## Участники и позиции"));
    const tr = translateTask({ lines: [seg({ startMs: 1_000, text: "қашан бастаймыз" })], speakers: SPEAKERS });
    expect(tr.text).toContain("<lines>\n[00:01] Участник 1: қашан бастаймыз\n</lines>");
  });

  it("picks Kazakh lines for translation, else the latest lines", () => {
    const finals = [
      seg({ id: "1", startMs: 1, text: "привет" }),
      seg({ id: "2", startMs: 2, text: "тестілеуді қашан бастаймыз" }),
      seg({ id: "3", startMs: 3, text: "в среду" }),
    ];
    expect(pickTranslationLines(finals, []).map((s) => s.id)).toEqual(["2"]);
    expect(pickTranslationLines(finals.filter((s) => s.id !== "2"), []).map((s) => s.id)).toEqual(["1", "3"]);
  });

  it("multi-party wording: rolling summary says who said what, tasks carry owners by name", () => {
    expect(rollingSummaryTask({ previous: null, speakers: SPEAKERS }).text).toContain("кто что сказал");
    expect(finalSummaryTask({ speakers: SPEAKERS }).text).toContain("«- **Кто** — что сделать — срок»");
  });

  it("every task that reads the transcript repeats the names map", () => {
    const named = new SpeakerDirectory("room", { "sys:3": "Айдос", "mic:1": "Мария" });
    const line = namesMapLine(named)!;
    expect(line).toBe("Имена участников: Участник 3 — Айдос; Зал 1 — Мария. Остальные подписаны метками по умолчанию.");
    expect(namesMapLine(SPEAKERS)).toBeNull();
    for (const t of [
      hintTask({ lang: "auto", speakers: named }),
      askTask({ question: "?", lang: "auto", speakers: named }),
      recapTask({ minutes: 5, nowMs: 0, speakers: named }),
      rollingSummaryTask({ previous: null, speakers: named }),
      finalSummaryTask({ speakers: named }),
    ]) {
      expect(t.text).toContain(line);
    }
  });

  it("addressed hints allow SKIP; named and silence hints say why they fired", () => {
    const q = seg({ text: "хуго а по срокам что", speaker: "sys:2" });
    const named = hintTask({ lang: "auto", speakers: SPEAKERS, question: q, auto: "named", skippable: true, myNames: ["Хуго"] });
    expect(named.skippable).toBe(true);
    expect(named.label).toBe("Обращаются к вам");
    expect(named.text).toContain("обратились по имени — Участник 2: «хуго а по срокам что»");
    expect(named.text).toContain("ответь ровно SKIP");
    expect(named.text).toContain("Ко мне обращаются так: Хуго.");
    const silence = hintTask({ lang: "auto", speakers: SPEAKERS, question: q, auto: "silence", skippable: true });
    expect(silence.text).toContain("все замолчали");
    expect(silence.text).toContain("ответь ровно SKIP");
    // Manual hints are never skippable, even if asked.
    expect(hintTask({ lang: "auto", speakers: SPEAKERS, skippable: true }).skippable).toBe(false);
  });

  it("recognizes SKIP replies, finished and while streaming", () => {
    for (const t of ["SKIP", " skip ", "**SKIP**", "SKIP.", "SKIP — вопрос к Айдосу", "«SKIP»"]) expect(isSkipReply(t)).toBe(true);
    for (const t of ["Скажите, что успеем", "SKIPPED", "", `SKIP ${"очень длинное объяснение ".repeat(10)}`]) {
      expect(isSkipReply(t)).toBe(false);
    }
    for (const t of ["", "S", "SK", "**SKI", "SKIP", "SKIP —"]) expect(mayBeSkip(t)).toBe(true);
    for (const t of ["Ск", "Sure", "- Предлагаю", "Да"]) expect(mayBeSkip(t)).toBe(false);
  });

  it("budgets: live and fork share hint effort, final uses summary effort", () => {
    expect(budgetFor("live", "low", "high")).toEqual({ effort: "low", maxTokens: 8_000 });
    expect(budgetFor("fork", "low", "high").effort).toBe("low");
    expect(budgetFor("final", "low", "high")).toEqual({ effort: "high", maxTokens: 32_000 });
    expect(budgetFor("final", "low", "xhigh").maxTokens).toBe(64_000);
  });
});
