import { describe, expect, it } from "vitest";
import type { Segment } from "../types";
import { SpeakerDirectory } from "./speakers";
import {
  MIN_SUGGESTION_CONFIDENCE,
  SUGGEST_REQUEST,
  SUGGEST_SYSTEM_PROMPT,
  SUGGESTION_SCHEMA,
  buildSuggestRequest,
  parseSuggestions,
  suggestTranscript,
  unnamedLabels,
  type SuggestInput,
} from "./suggestions";

function seg(id: string, speaker: string | null, startMs: number, text: string): Segment {
  return { id, source: speaker?.startsWith("mic") || speaker === "me" ? "mic" : "system", speaker, startMs, endMs: startMs + 2_000, text, isFinal: true };
}

const FINALS = [
  seg("s1", "sys:1", 1_000, "айдос что по мобилке"),
  seg("s2", "sys:3", 4_000, "выкатили на десять процентов"),
  seg("s3", "sys:1", 8_000, "спасибо нурлан"),
  seg("m1", "me", 9_000, "да я возьму"),
  seg("s4", "sys:2", 10_000, "второй брокер поднимем"),
  seg("s5", null, 12_000, "ок"),
];

function input(names: Record<string, string> = { "sys:1": "Айгерим" }): SuggestInput {
  return { finals: FINALS, speakers: new SpeakerDirectory("me", names), myNames: ["Хуго"], rejected: [{ label: "sys:2", name: "Ерлан" }] };
}

describe("speaker-name suggestions: request", () => {
  it("lists only renamable labels without a name", () => {
    expect(unnamedLabels(FINALS, input().speakers)).toEqual(["sys:2", "sys:3"]);
  });

  it("is a small separate request: own system prompt, one user message, no cache breakpoints", () => {
    const p = buildSuggestRequest(input());
    expect(p.system).toEqual([{ type: "text", text: SUGGEST_SYSTEM_PROMPT }]);
    expect(p.messages).toHaveLength(1);
    expect(JSON.stringify(p)).not.toContain("cache_control");
    const text = (p.messages[0].content as Array<{ text: string }>)[0].text;
    expect(text).toContain("<unnamed_labels>\nsys:2 (сейчас «Участник 2»)\nsys:3 (сейчас «Участник 3»)\n</unnamed_labels>");
    expect(text).toContain("<named_labels>\nsys:1 = Айгерим\n</named_labels>");
    expect(text).toContain("к пользователю обращаются: Хуго");
    expect(text).toContain("<rejected>\nsys:2 — не Ерлан\n</rejected>");
    expect(text).toContain("[00:04] sys:3 (Участник 3): выкатили на десять процентов");
    expect(text).toContain("[00:12] ? (?): ок");
    expect(text).toContain("[00:09] me (Я): да я возьму");
  });

  it("keeps only the recent end of a long transcript", () => {
    const long = Array.from({ length: 200 }, (_, i) => seg(`x${i}`, "sys:1", i * 1_000, `реплика номер ${i}`));
    const t = suggestTranscript(long, new SpeakerDirectory("me"));
    const lines = t.split("\n");
    expect(lines.length).toBeLessThanOrEqual(80);
    expect(lines.at(-1)).toContain("реплика номер 199");
  });

  it("asks for low effort, JSON-schema structured output and no server fallbacks", () => {
    expect(SUGGEST_REQUEST).toMatchObject({ effort: "low", fallbacks: false, format: SUGGESTION_SCHEMA });
    expect(SUGGESTION_SCHEMA.type).toBe("json_schema");
    const schema = SUGGESTION_SCHEMA.schema as { additionalProperties: boolean; properties: { suggestions: { items: { required: string[]; additionalProperties: boolean } } } };
    expect(schema.additionalProperties).toBe(false);
    expect(schema.properties.suggestions.items.required).toEqual(["label", "name", "evidence", "confidence"]);
    expect(schema.properties.suggestions.items.additionalProperties).toBe(false);
    // Numeric/length constraints are not supported by structured outputs.
    expect(JSON.stringify(schema)).not.toMatch(/minimum|maximum|minLength|maxLength/);
  });
});

describe("speaker-name suggestions: parsing", () => {
  it("keeps valid suggestions for unnamed labels, one per label, best confidence", () => {
    const reply = JSON.stringify({
      suggestions: [
        { label: "sys:3", name: "Айдос", evidence: "«айдос что по мобилке» — и отвечает sys:3", confidence: 0.82 },
        { label: "sys:3", name: "Айдар", evidence: "похоже", confidence: 0.65 },
        { label: "sys:2", name: "Нурлан", evidence: "«спасибо нурлан» после его реплики", confidence: 0.7 },
      ],
    });
    expect(parseSuggestions(reply, input())).toEqual([
      { label: "sys:2", name: "Нурлан", evidence: "«спасибо нурлан» после его реплики", confidence: 0.7 },
      { label: "sys:3", name: "Айдос", evidence: "«айдос что по мобилке» — и отвечает sys:3", confidence: 0.82 },
    ]);
  });

  it("drops named, unknown and own labels, low confidence, rejected pairs and junk names", () => {
    const reply = JSON.stringify({
      suggestions: [
        { label: "sys:1", name: "Айгерим", evidence: "", confidence: 0.9 }, // already named
        { label: "sys:9", name: "Кто-то", evidence: "", confidence: 0.9 }, // not in the meeting
        { label: "me", name: "Хуго", evidence: "", confidence: 0.9 },
        { label: "sys:3", name: "Айдос", evidence: "", confidence: MIN_SUGGESTION_CONFIDENCE - 0.01 },
        { label: "sys:2", name: "ерлан", evidence: "", confidence: 0.9 }, // rejected earlier (case-insensitive)
        { label: "sys:2", name: "Участник 2", evidence: "", confidence: 0.9 },
        { label: "sys:2", name: "<b>x</b>", evidence: "", confidence: 0.9 },
        { label: "sys:3", name: "Айдос", evidence: "", confidence: "0.9" },
        "garbage",
      ],
    });
    expect(parseSuggestions(reply, input())).toEqual([]);
  });

  it("clamps confidence and survives non-JSON wrappers (models without structured outputs)", () => {
    const reply = 'Вот ответ:\n```json\n{"suggestions":[{"label":"sys:3","name":"Айдос","evidence":"ответил на обращение","confidence":1.4}]}\n```';
    expect(parseSuggestions(reply, input())).toEqual([{ label: "sys:3", name: "Айдос", evidence: "ответил на обращение", confidence: 1 }]);
    expect(parseSuggestions("не знаю", input())).toEqual([]);
    expect(parseSuggestions('{"suggestions": 5}', input())).toEqual([]);
  });
});
