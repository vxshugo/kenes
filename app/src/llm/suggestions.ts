import type { Segment } from "../types";
import type { JsonSchemaFormat } from "./client";
import type { RequestPayload } from "./conversation";
import { cleanText, compareSegments, formatClock } from "./prompts";
import { compareLabels, defaultSpeakerName, isRenamable, type SpeakerDirectory } from "./speakers";

/**
 * Speaker-name suggestions: a small side request, separate from the meeting conversation,
 * asking which unnamed labels can be named from the dialogue itself ("Айдос, что скажешь?"
 * and sys:3 answers). It never touches the main log, so the live prompt cache is unaffected.
 */

export type NameSuggestion = {
  label: string;
  name: string;
  /** Short quote or reasoning from the dialogue. */
  evidence: string;
  /** 0..1 */
  confidence: number;
};

/** Suggestions below this confidence are not shown. */
export const MIN_SUGGESTION_CONFIDENCE = 0.6;
/** How much of the recent transcript the request carries. */
export const SUGGEST_MAX_LINES = 80;
export const SUGGEST_MAX_CHARS = 9_000;

/** Frozen instructions for the side request. Small on purpose; not cached. */
export const SUGGEST_SYSTEM_PROMPT = `Ты помогаешь подписать говорящих в расшифровке деловой встречи. Голоса разделены автоматически, у каждого метка: sys:N — голос из звонка, mic:N — голос в зале, me — сам пользователь. Расшифровка — сырой вывод распознавания речи: строчные буквы, без пунктуации, русский и казахский вперемешку, имена могут быть искажены.

Определи имена безымянных меток, только если имя следует из самого диалога:
- к человеку обратились по имени («айдос что скажешь», «айдосқа сұрақ»), и сразу после этого отвечает эта метка;
- говорящий представился («это дина из тестирования», «мен нұрланмын»);
- кто-то называет его по имени сразу после его реплики («спасибо айдос»).
Не угадывай по должностям, темам или повестке. Если к человеку обратились, а ответила другая метка или ответа не было — это не доказательство. Одно слабое совпадение — низкая уверенность.

Имя пиши в обычном написании с заглавной буквы, исправляя явные ошибки распознавания («айдос» → «Айдос», «нұрлан» → «Нұрлан»). Пользователя зовут так, как указано во входных данных; если по диалогу видно, что безымянная метка — это он сам, предложи его имя.

Ответ — только JSON: {"suggestions": [{"label": "sys:3", "name": "Айдос", "evidence": "короткая цитата или пояснение по-русски", "confidence": 0.0–1.0}]}. Если назвать некого — {"suggestions": []}.`;

export const SUGGESTION_SCHEMA: JsonSchemaFormat = {
  type: "json_schema",
  schema: {
    type: "object",
    properties: {
      suggestions: {
        type: "array",
        items: {
          type: "object",
          properties: {
            label: { type: "string", description: "Метка говорящего из списка безымянных, например sys:3" },
            name: { type: "string", description: "Имя в обычном написании" },
            evidence: { type: "string", description: "Короткая цитата или пояснение, почему это он" },
            confidence: { type: "number", description: "Уверенность от 0 до 1" },
          },
          required: ["label", "name", "evidence", "confidence"],
          additionalProperties: false,
        },
      },
    },
    required: ["suggestions"],
    additionalProperties: false,
  },
};

export type SuggestInput = {
  finals: readonly Segment[];
  speakers: SpeakerDirectory;
  /** How the user is addressed (Settings.myNames). */
  myNames: readonly string[];
  /** Pairs the user already rejected; not suggested again. */
  rejected: ReadonlyArray<{ label: string; name: string }>;
};

/** Labels that spoke, can be named and have no name yet, in label order. */
export function unnamedLabels(finals: readonly Segment[], speakers: SpeakerDirectory): string[] {
  const seen = new Set<string>();
  for (const f of finals) {
    const label = speakers.labelOf(f);
    if (isRenamable(label) && !speakers.hasCustomName(label)) seen.add(label);
  }
  return [...seen].sort(compareLabels);
}

/** The recent part of the transcript, one line per segment with an explicit label. */
export function suggestTranscript(finals: readonly Segment[], speakers: SpeakerDirectory): string {
  const lines = [...finals]
    .filter((s) => cleanText(s.text))
    .sort(compareSegments)
    .map((s) => {
      const label = speakers.labelOf(s);
      return `[${formatClock(s.startMs)}] ${label ?? "?"} (${speakers.nameOf(label)}): ${cleanText(s.text)}`;
    });
  const out: string[] = [];
  let size = 0;
  for (let i = lines.length - 1; i >= 0 && out.length < SUGGEST_MAX_LINES; i--) {
    size += lines[i].length + 1;
    if (size > SUGGEST_MAX_CHARS && out.length) break;
    out.unshift(lines[i]);
  }
  return out.join("\n");
}

/** The side request: its own tiny system prompt, one user message, no cache breakpoints. */
export function buildSuggestRequest(input: SuggestInput): RequestPayload {
  const unnamed = unnamedLabels(input.finals, input.speakers);
  const named = Object.entries(input.speakers.snapshot());
  const myNames = input.myNames.map(cleanText).filter(Boolean);
  const text = [
    "<unnamed_labels>",
    unnamed.map((l) => `${l} (сейчас «${defaultSpeakerName(l)}»)`).join("\n") || "(нет)",
    "</unnamed_labels>",
    named.length ? `<named_labels>\n${named.map(([l, n]) => `${l} = ${n}`).join("\n")}\n</named_labels>` : null,
    `<user>метка me; ${myNames.length ? `к пользователю обращаются: ${myNames.join(", ")}` : "имя пользователя не указано"}</user>`,
    input.rejected.length
      ? `<rejected>\n${input.rejected.map((r) => `${r.label} — не ${r.name}`).join("\n")}\n</rejected>`
      : null,
    "<transcript>",
    suggestTranscript(input.finals, input.speakers) || "(пусто)",
    "</transcript>",
    "Для каких безымянных меток имя следует из диалога? Ответь JSON по схеме.",
  ]
    .filter((l): l is string => typeof l === "string")
    .join("\n");
  return {
    system: [{ type: "text", text: SUGGEST_SYSTEM_PROMPT }],
    messages: [{ role: "user", content: [{ type: "text", text }] }],
  };
}

function extractJson(text: string): unknown {
  const t = text.trim();
  try {
    return JSON.parse(t);
  } catch {
    // Plain-text fallback (model without structured outputs): take the outermost object.
    const a = t.indexOf("{");
    const b = t.lastIndexOf("}");
    if (a < 0 || b <= a) return null;
    try {
      return JSON.parse(t.slice(a, b + 1));
    } catch {
      return null;
    }
  }
}

const NAME_OK = /^[\p{L}][\p{L}\p{M}\s.'’-]{0,39}$/u;

/**
 * Validates the reply against what was asked: known unnamed labels only, a plausible name,
 * confidence ≥ `MIN_SUGGESTION_CONFIDENCE`, not a rejected pair; one suggestion per label.
 */
export function parseSuggestions(text: string, input: SuggestInput): NameSuggestion[] {
  const data = extractJson(text) as { suggestions?: unknown } | null;
  const list = Array.isArray(data?.suggestions) ? (data!.suggestions as unknown[]) : [];
  const allowed = new Set(unnamedLabels(input.finals, input.speakers));
  const rejected = new Set(input.rejected.map((r) => `${r.label}\u0000${r.name.toLowerCase()}`));
  const best = new Map<string, NameSuggestion>();
  for (const raw of list) {
    if (!raw || typeof raw !== "object") continue;
    const r = raw as Record<string, unknown>;
    const label = typeof r.label === "string" ? r.label.trim() : "";
    const name = typeof r.name === "string" ? cleanText(r.name) : "";
    const evidence = typeof r.evidence === "string" ? cleanText(r.evidence).slice(0, 200) : "";
    const confidence = typeof r.confidence === "number" && Number.isFinite(r.confidence) ? Math.min(1, Math.max(0, r.confidence)) : 0;
    if (!allowed.has(label) || !NAME_OK.test(name) || name === defaultSpeakerName(label)) continue;
    if (confidence < MIN_SUGGESTION_CONFIDENCE || rejected.has(`${label}\u0000${name.toLowerCase()}`)) continue;
    const prev = best.get(label);
    if (!prev || confidence > prev.confidence) best.set(label, { label, name, evidence, confidence });
  }
  return [...best.values()].sort((a, b) => compareLabels(a.label, b.label));
}

/** Request settings: cheap and quick, no server-side fallbacks (a refused suggestion is just dropped). */
export const SUGGEST_REQUEST = {
  effort: "low" as const,
  maxTokens: 4_000,
  timeoutMs: 60_000,
  maxRetries: 1,
  fallbacks: false,
  format: SUGGESTION_SCHEMA,
};
