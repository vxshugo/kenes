import type { AnswerLanguage, HintEffort, Segment, SummaryEffort } from "../types";
import {
  cleanText,
  compareSegments,
  formatClock,
  formatSegmentLine,
  hasKazakhLetters,
  languageInstruction,
} from "./prompts";
import { defaultSpeakerName, type SpeakerDirectory } from "./speakers";

/**
 * Where a task runs:
 * - "live": appended to the meeting conversation (serialized, append-only, cached prefix).
 * - "fork": reads the conversation's cached prefix but is not appended to it (rolling summary).
 * - "final": a separate request with the full transcript (final summary, at summaryEffort).
 */
export type TaskRoute = "live" | "fork" | "final";

export type TaskKind = "hint" | "ask" | "explain" | "translate" | "recap" | "rolling" | "final";

export type TaskSpec = {
  kind: TaskKind;
  route: TaskRoute;
  /** The `<task>` text appended after the transcript update. */
  text: string;
  /** Short Russian label for the UI card. */
  label: string;
  /** Saved with the note (`save_note.trigger`). */
  trigger: string | null;
  /** Claude may answer exactly SKIP (the question wasn't for the user); the UI then drops the card. */
  skippable?: boolean;
};

/** Said to reduce thinking before the first visible token on latency-sensitive routes. */
const LATENCY_LINE = "Latency-sensitive; begin your visible answer immediately.";

export const SKIP_TOKEN = "SKIP";

const SKIP_LINE = `Если вопрос или реплика адресованы не мне, а другому участнику, ответь ровно ${SKIP_TOKEN} — одно слово, без пояснений.`;

function task(type: string, lines: Array<string | null | undefined | false>): string {
  const body = lines.filter((l): l is string => typeof l === "string" && l.length > 0).join("\n");
  return `<task type="${type}">\n${body}\n</task>`;
}

function quote(text: string, max = 240): string {
  const t = cleanText(text);
  return t.length > max ? `${t.slice(0, max - 1)}…` : t;
}

/**
 * The current names map, repeated in every task so Claude never has to reconstruct it from
 * rename notes scattered through the log. Null when nobody has been renamed.
 */
export function namesMapLine(speakers: SpeakerDirectory): string | null {
  const entries = Object.entries(speakers.snapshot());
  if (!entries.length) return null;
  const pairs = entries.map(([label, name]) => `${defaultSpeakerName(label)} — ${name}`);
  return `Имена участников: ${pairs.join("; ")}. Остальные подписаны метками по умолчанию.`;
}

function myNamesLine(myNames: readonly string[] | undefined): string | null {
  const names = (myNames ?? []).map(cleanText).filter(Boolean);
  return names.length ? `Ко мне обращаются так: ${names.join(", ")}.` : null;
}

function inProgressBlock(partials: readonly Segment[], speakers: SpeakerDirectory): string | null {
  const live = partials.filter((p) => cleanText(p.text));
  if (!live.length) return null;
  const lines = [...live].sort(compareSegments).map((p) => `${formatSegmentLine(p, speakers)} …`);
  return `Ещё звучит (распознавание не завершено, говорящий может быть не определён):\n${lines.join("\n")}`;
}

/**
 * Why an automatic hint fired:
 * - "any": a question from someone else (auto-hint mode "any");
 * - "named": the user was addressed by name (mode "addressed");
 * - "silence": a question, then silence, without the user's name (mode "addressed").
 */
export type AutoKind = "any" | "named" | "silence";

export function hintTask(opts: {
  lang: AnswerLanguage;
  speakers: SpeakerDirectory;
  /** The segment that fired an automatic hint. */
  question?: Segment | null;
  auto?: AutoKind;
  /** Allow a SKIP answer (auto-hint mode "addressed"). */
  skippable?: boolean;
  myNames?: readonly string[];
  partials?: readonly Segment[];
}): TaskSpec {
  const q = opts.question ? quote(opts.question.text) : null;
  const auto = !!q;
  const who = opts.question ? opts.speakers.nameFor(opts.question) : null;
  const said = q ? `${who}: «${q}»` : null;
  const skippable = auto && !!opts.skippable;
  let lead: string;
  if (!auto) {
    lead =
      "Что мне сейчас ответить? Отвечай на последний вопрос или реплику, обращённую ко мне; если явного вопроса нет — подскажи, что уместно сказать дальше.";
  } else if (opts.auto === "named") {
    lead = `Похоже, ко мне обратились по имени — ${said}. Что мне ответить?`;
  } else if (opts.auto === "silence") {
    lead = `Прозвучал вопрос, и после него все замолчали — ${said}. Меня по имени не называли. Реши по ходу разговора, ждут ли ответа от меня (вопрос ко всем, по моей теме, продолжение моего разговора). Если да — что мне ответить?`;
  } else {
    lead = `Похоже, прозвучал вопрос — ${said}. Что мне ответить?`;
  }
  return {
    kind: "hint",
    route: "live",
    label: !auto ? "Что ответить?" : opts.auto === "named" ? "Обращаются к вам" : "Подсказка (авто)",
    trigger: auto && q ? `auto: ${q}` : "manual",
    skippable,
    text: task("hint", [
      lead,
      skippable ? SKIP_LINE : null,
      inProgressBlock(opts.partials ?? [], opts.speakers),
      "Дай подсказку, которую можно сразу произнести: до 5 коротких пунктов или 2–3 предложения. Опирайся на контекст встречи и сказанное; если нужных фактов нет — нейтральная формулировка и строка «Уточнить: …».",
      myNamesLine(opts.myNames),
      namesMapLine(opts.speakers),
      languageInstruction(opts.lang),
      LATENCY_LINE,
    ]),
  };
}

/**
 * True when a finished reply is the SKIP answer. Tolerates markdown or punctuation around
 * it and a short explanation after it, which the prompt forbids but models occasionally add.
 */
export function isSkipReply(text: string): boolean {
  const t = text.trim().replace(/^[\s*_`"'«»“”.,:;!-]+/, "");
  if (!/^skip\b/i.test(t)) return false;
  return t.length <= 160;
}

/** While streaming: could this text still turn out to be SKIP? Such a card stays hidden. */
export function mayBeSkip(partial: string): boolean {
  const t = partial.trim().replace(/^[\s*_`"'«»“”.,:;!-]+/, "").toUpperCase();
  if (!t) return true;
  return SKIP_TOKEN.startsWith(t) || t.startsWith(SKIP_TOKEN);
}

export function askTask(opts: { question: string; lang: AnswerLanguage; speakers: SpeakerDirectory; partials?: readonly Segment[] }): TaskSpec {
  const q = cleanText(opts.question);
  return {
    kind: "ask",
    route: "live",
    label: "Вопрос ассистенту",
    trigger: `ask: ${quote(q, 200)}`,
    text: task("ask", [
      `Мой вопрос тебе (не участникам встречи): «${q}»`,
      inProgressBlock(opts.partials ?? [], opts.speakers),
      "Ответь по существу и коротко, опираясь на встречу и контекст. Если ответа в них нет — скажи прямо и предложи, как уточнить.",
      namesMapLine(opts.speakers),
      opts.lang === "kk"
        ? "Язык ответа: казахский, с коротким переводом на русском."
        : "Язык ответа: русский (казахские цитаты — с переводом).",
      LATENCY_LINE,
    ]),
  };
}

export function explainTask(opts: { term: string; lang: AnswerLanguage }): TaskSpec {
  const term = quote(opts.term, 300);
  return {
    kind: "explain",
    route: "live",
    label: "Объяснение",
    trigger: `explain: ${term}`,
    text: task("explain", [
      `Объясни «${term}» простыми словами в контексте этой встречи.`,
      "Формат: 1–2 предложения, что это и почему это здесь важно; если уместно — одна готовая фраза, как это использовать в разговоре. Если это казахское слово или фраза — сначала перевод. Если похоже на ошибку распознавания — скажи, что, вероятно, имелось в виду.",
      opts.lang === "kk"
        ? "Язык ответа: казахский, с коротким переводом на русском."
        : "Язык ответа: русский.",
      LATENCY_LINE,
    ]),
  };
}

/** Recent lines worth translating: Kazakh or mixed ones among the last `window` finals, plus partials. */
export function pickTranslationLines(
  finals: readonly Segment[],
  partials: readonly Segment[],
  window = 20,
): Segment[] {
  const recent = [...finals].sort(compareSegments).slice(-window);
  const pool = [...recent, ...partials.filter((p) => cleanText(p.text))];
  const kazakh = pool.filter((s) => hasKazakhLetters(s.text));
  return kazakh.length ? kazakh.slice(-10) : pool.slice(-6);
}

export function translateTask(opts: { lines: readonly Segment[]; speakers: SpeakerDirectory }): TaskSpec {
  const rendered = opts.lines.map((s) => formatSegmentLine(s, opts.speakers)).join("\n");
  return {
    kind: "translate",
    route: "live",
    label: "Перевод",
    trigger: "translate",
    text: task("translate", [
      "Переведи на русский казахские и смешанные реплики из этих строк:",
      `<lines>\n${rendered || "(нет строк)"}\n</lines>`,
      "Формат: по строке на реплику — «[мм:сс] Говорящий: перевод». Чисто русские реплики пропускай. Если казахского нет — ответь одной строкой, что переводить нечего. Ошибки распознавания исправляй по смыслу.",
      LATENCY_LINE,
    ]),
  };
}

export function recapTask(opts: { minutes: number; nowMs: number; speakers: SpeakerDirectory }): TaskSpec {
  const from = Math.max(0, opts.nowMs - opts.minutes * 60_000);
  return {
    kind: "recap",
    route: "live",
    label: `Кратко: ${opts.minutes} мин`,
    trigger: `recap: ${opts.minutes}m`,
    text: task("recap", [
      `Кратко перескажи, что обсуждалось за последние ${opts.minutes} минут (примерно с [${formatClock(from)}]).`,
      "3–5 пунктов: о чём говорили, кто что предложил или возразил (по именам), что решили, какие вопросы задали мне и что я пообещал. Без вступлений.",
      namesMapLine(opts.speakers),
      "Язык ответа: русский.",
      LATENCY_LINE,
    ]),
  };
}

export function rollingSummaryTask(opts: { previous: string | null; speakers: SpeakerDirectory }): TaskSpec {
  const prev = opts.previous?.trim();
  return {
    kind: "rolling",
    route: "fork",
    label: "Резюме по ходу встречи",
    trigger: "rolling",
    text: task("rolling_summary", [
      "Составь краткое резюме встречи на текущий момент, чтобы я за 10 секунд вспомнил ход разговора.",
      prev ? `Прошлое резюме (обнови его, а не пиши с нуля):\n<previous_summary>\n${prev}\n</previous_summary>` : null,
      "Формат Markdown без заголовков: первая строка — «**Главное:** …» (1–2 предложения), дальше до 7 пунктов: темы, договорённости (кто, что, срок), открытые вопросы, вопросы ко мне без ответа.",
      "Участников много: когда важно, кто что сказал, предложил или возразил, называй человека («Айдос против переноса»). Если имени нет, используй его метку («Участник 3»).",
      namesMapLine(opts.speakers),
      "Язык: русский. Только то, что прозвучало.",
    ]),
  };
}

export function finalSummaryTask(opts: { speakers: SpeakerDirectory }): TaskSpec {
  return {
    kind: "final",
    route: "final",
    label: "Итоги встречи",
    trigger: "final",
    text: task("final_summary", [
      "Встреча закончилась. Подготовь итоговое резюме по всей расшифровке и контексту встречи. В ней участвовало много людей — важно, кто что сказал и кто за что отвечает.",
      "Формат — Markdown, по-русски, ровно эти разделы в этом порядке:",
      "## Итоги\n2–4 предложения: о чём была встреча и чем закончилась.",
      "## Решения\nСписок принятых решений.",
      "## Задачи\nСписок в формате «- **Кто** — что сделать — срок». «Кто» — имя участника, как он подписан в расшифровке (или его метка вроде «Участник 3», если имени нет); мои задачи — «**Я**». Если ответственный или срок не прозвучали, пиши «не назван».",
      "## Участники и позиции\nПо строке на каждого, кто высказался по существу: «- **Имя** — ключевые тезисы, предложения, возражения, обещания». Себя — «**Я**». Сначала те, кто повлиял на решения. Тех, кто только поздоровался или поддакнул, пропускай.",
      "## Открытые вопросы\nЧто осталось без ответа или требует уточнения, и к кому вопрос.",
      "## Ключевые цифры\nЧисла, суммы, даты, метрики с пояснением, к чему они относятся.",
      namesMapLine(opts.speakers),
      "Если в разделе нечего написать — «—». Без таблиц. Не выдумывай: только то, что прозвучало или есть в контексте. Казахские фрагменты передавай по-русски. Объём — сколько нужно по существу, без воды и повторов.",
    ]),
  };
}

export type RequestBudget = { maxTokens: number; effort: HintEffort | SummaryEffort };

/**
 * `max_tokens` caps thinking + text together (adaptive thinking is on), so leave room.
 * At xhigh the docs recommend starting from 64K.
 */
export function budgetFor(route: TaskRoute, hintEffort: HintEffort, summaryEffort: SummaryEffort): RequestBudget {
  if (route === "final") {
    return { effort: summaryEffort, maxTokens: summaryEffort === "xhigh" ? 64_000 : 32_000 };
  }
  // Live turns and the rolling-summary fork share one effort so they share one cache.
  const maxTokens = hintEffort === "high" ? 16_000 : hintEffort === "medium" ? 12_000 : 8_000;
  return { effort: hintEffort, maxTokens: route === "fork" ? Math.max(maxTokens, 12_000) : maxTokens };
}
