import type { AutoHintMode, Segment } from "../types";

/**
 * Lexical question detection for raw ASR output (lowercase, no punctuation, Russian and
 * Kazakh mixed). With no "?" to rely on we look for request verbs, question phrases and
 * particles, and interrogative words in a question position.
 */

const KAZAKH_LETTERS = /[әғқңөұүһі]/;

/** Tokens: lowercase words (ё folded to е); hyphens and punctuation split words. */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/ё/g, "е")
    .split(/[^a-zа-яәғқңөұүһі0-9]+/)
    .filter(Boolean);
}

/**
 * Russian interrogatives that double as conjunctions/relatives ("я думаю что…",
 * "когда мы запустили…"): count only at the start of the utterance.
 */
const RU_WH_AMBIGUOUS = new Set(["как", "что", "чего", "чем", "когда", "где", "кто", "кого", "кому", "кем"]);

/** Russian interrogatives that are questions almost anywhere ("а по возвратам какие риски видите"). */
const RU_WH_STRONG = new Set([
  "почему", "зачем", "отчего", "сколько", "насколько", "куда", "откуда",
  "какой", "какая", "какое", "какие", "каким", "каких", "какую", "какого", "какому", "какими",
  "каков", "какова", "каковы", "чей", "чья", "чье", "чьи",
]);

/** Kazakh interrogatives; Kazakh puts them late in the clause, so any position counts. */
const KK_WH = new Set([
  "қалай", "қалайша", "неге", "неліктен", "несіне", "қашан", "қайда", "қайдан", "қайсы",
  "қайсысы", "кім", "кімге", "кімнің", "кімді", "кіммен", "қанша", "қандай", "неше", "нешеу",
  "нешінші", "нені", "неден", "немен",
]);

/** Polite imperatives asking the listener to talk: as good as a question. */
const REQUEST_VERBS = new Set([
  "расскажите", "расскажи", "объясните", "объясни", "поясните", "поясни", "уточните",
  "подскажите", "подскажи", "скажите", "скажи", "опишите", "прокомментируйте", "покажите",
  "поделитесь", "назовите", "перечислите", "напомните",
  "айтыңызшы", "айтыңыз", "айтшы", "айтып", "түсіндіріңізші", "түсіндіріңіз", "түсіндірші",
  "түсіндіріп", "көрсетіңізші", "көрсетіңіз",
]);

/** Kazakh yes/no question particles written as separate words. */
const KK_PARTICLES = new Set(["ма", "ме", "ба", "бе", "па", "пе"]);

/** Kazakh particles glued to the word by the recognizer: "барма", "білесізбе", "дайынсызба". */
const KK_GLUED =
  /^(?:бар|жоқ|бола|болады|болды|білесіз|келесіз|келісесіз|дайынсыз|түсінікті|рас|дұрыс|мүмкін|керек)(?:ма|ме|ба|бе|па|пе)$|(?:сыз|сіз|сыздар|сіздер)(?:ба|бе|па|пе)$/;

/** Words after which a separate particle is Kazakh even without Kazakh-only letters. */
const KK_PARTICLE_HOSTS = new Set(["бар", "керек", "болады", "бола", "рас", "дайынсыз", "келесіз", "білесіз", "мүмкін", "жоқ"]);

/** Multi-word markers, matched on whole tokens. */
const PHRASES = [
  "можно ли", "могли бы", "не могли бы", "можете ли", "не подскажете", "как вы думаете",
  "как вы считаете", "что вы думаете", "что думаете", "что скажете", "ваше мнение",
  "как считаете", "а вы", "а у вас", "а вам", "а ты", "а у тебя", "или нет", "да или нет",
  "согласны ли", "вы согласны", "не так ли", "не правда ли", "как насчет",
  "сіз ше", "сіздер ше", "сендер ше", "сен ше", "не үшін", "не дейсіз", "қалай ойлайсыз",
];

/** Kazakh greetings that are literally questions ("are you well?"): drop before analysis. */
const GREETINGS = [
  "сәлеметсіз бе", "сәлеметсіздер ме", "сәлеметсіздер бе", "амансыз ба", "амансыздар ма",
  "қалыңыз қалай", "қалың қалай", "как дела", "как ваши дела",
];

/** Before a wh-word these make it a conjunction: "так как", "потому что", "вот почему", "не знаю почему". */
const WH_BLOCKERS_BEFORE = new Set([
  "так", "потому", "то", "чтобы", "также", "тогда", "после", "пока", "вот", "знаю", "знаем",
  "понятно", "ясно", "неважно", "независимо", "сказал", "сказали", "объяснил", "понимаю", "поэтому",
]);
/** After a wh-word these make it indefinite or a filler: "что то", "как бы", "как раз", "что касается". */
const WH_BLOCKERS_AFTER = new Set([
  "то", "нибудь", "либо", "бы", "раз", "только", "минимум", "максимум", "обычно", "всегда",
  "правило", "будто", "ни", "касается", "угодно", "попало",
]);
/** "вряд ли", "едва ли", "то ли … то ли": not questions. */
const LI_BLOCKERS = new Set(["вряд", "навряд", "едва", "то"]);

const FILLERS = new Set([
  "а", "и", "ну", "так", "вот", "слушайте", "коллеги", "ал", "енді", "иә", "жарайды", "ок",
  "окей", "хорошо", "понятно", "ясно", "да", "ладно", "тогда", "кстати", "еще", "ещё",
]);

const PRONOUNS = new Set(["я", "мы", "он", "она", "они", "оно"]);

const ADDRESS = new Set([
  "вы", "вас", "вам", "вами", "ваш", "ваша", "ваше", "ваши", "вашей", "вашего", "вашу", "вашим",
  "ты", "тебя", "тебе", "твой", "твоя", "твои",
  "сіз", "сізге", "сізде", "сіздің", "сізді", "сізбен", "сіздер", "сендер", "сен", "саған", "сенде", "сенің",
]);

/** 2nd-person / 1st-plural Kazakh verb endings: "не ойлайсыз", "не істейміз". */
const KK_VERBISH = /(сыз|сіз|сыздар|сіздер|мыз|міз|ймыз|йміз|сың|сің)$/;

export type QuestionAnalysis = {
  isQuestion: boolean;
  /** Which rule matched, for debugging and tests. */
  reason: string | null;
  words: number;
};

function stripGreetings(tokens: string[]): string[] {
  let joined = ` ${tokens.join(" ")} `;
  for (const g of GREETINGS) joined = joined.split(` ${g} `).join(" ");
  return joined.trim() ? joined.trim().split(" ") : [];
}

function blocked(tokens: string[], i: number): boolean {
  const next = tokens[i + 1];
  const prev = tokens[i - 1];
  return (next !== undefined && WH_BLOCKERS_AFTER.has(next)) || (prev !== undefined && WH_BLOCKERS_BEFORE.has(prev));
}

/** Kazakh "не" = "what", but in Russian "не" = "not"; accept it only in Kazakh-looking context. */
function isKazakhWhat(tokens: string[], i: number, kazakhish: boolean): boolean {
  if (tokens[i] !== "не" || !kazakhish) return false;
  const next = tokens[i + 1];
  return next === undefined || KAZAKH_LETTERS.test(next) || KK_VERBISH.test(next) || next === "бар";
}

/** Decides whether a (final) utterance looks like a question. */
export function analyzeQuestion(text: string): QuestionAnalysis {
  const raw = tokenize(text);
  const words = raw.length;
  const tokens = stripGreetings(raw);
  const no = (reason: string | null = null): QuestionAnalysis => ({ isQuestion: false, reason, words });
  const yes = (reason: string): QuestionAnalysis => ({ isQuestion: true, reason, words });
  if (!tokens.length) return no();

  const joined = ` ${tokens.join(" ")} `;
  const kazakhish = tokens.some((t) => KAZAKH_LETTERS.test(t));
  const last = tokens.length - 1;

  for (const p of PHRASES) {
    if (joined.includes(` ${p} `)) return yes(`phrase:${p}`);
  }
  for (const t of tokens) {
    if (REQUEST_VERBS.has(t)) return yes(`request:${t}`);
  }

  // Russian particle "ли": "готовы ли вы", "успеваете ли к пятнице".
  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i] === "ли" && !LI_BLOCKERS.has(tokens[i - 1])) return yes("particle:ли");
  }

  // Kazakh particles: "бар ма", "келісесіз бе", "керек пе әлде …".
  for (let i = 1; i < tokens.length; i++) {
    if (!KK_PARTICLES.has(tokens[i])) continue;
    const prev = tokens[i - 1];
    const next = tokens[i + 1];
    if (kazakhish || i === last || KK_PARTICLE_HOSTS.has(prev) || next === "екен" || next === "әлде") {
      return yes(`particle:${tokens[i]}`);
    }
  }
  for (const t of tokens) {
    if (KK_GLUED.test(t)) return yes(`glued:${t}`);
  }
  if (kazakhish && tokens[last] === "ше" && tokens.length > 1) return yes("particle:ше");

  for (let i = 0; i < tokens.length; i++) {
    if (KK_WH.has(tokens[i])) return yes(`wh-kk:${tokens[i]}`);
    if (isKazakhWhat(tokens, i, kazakhish)) return yes("wh-kk:не");
  }

  for (let i = 0; i < tokens.length; i++) {
    if (RU_WH_STRONG.has(tokens[i]) && !blocked(tokens, i)) return yes(`wh:${tokens[i]}`);
  }

  // Ambiguous interrogatives: only near the start, after optional fillers.
  let start = 0;
  while (start < Math.min(3, last) && FILLERS.has(tokens[start])) start++;
  for (let i = start; i < Math.min(tokens.length, start + 2); i++) {
    const t = tokens[i];
    if (!RU_WH_AMBIGUOUS.has(t) || blocked(tokens, i)) continue;
    // "когда мы запустили первую версию было много багов": a long subordinate clause.
    const next = tokens[i + 1];
    const addressed = tokens.some((x) => ADDRESS.has(x));
    if (next && PRONOUNS.has(next) && tokens.length >= 7 && !addressed) continue;
    return yes(`wh-start:${t}`);
  }

  // "… последний вопрос … кто будет делать": an announced question.
  if (tokens.some((t) => t === "вопрос" || t === "вопросик" || t === "сұрақ")) {
    for (let i = 0; i < tokens.length; i++) {
      if (RU_WH_AMBIGUOUS.has(tokens[i]) && !blocked(tokens, i)) return yes(`wh-announced:${tokens[i]}`);
    }
  }
  return no();
}

export function looksLikeQuestion(text: string): boolean {
  return analyzeQuestion(text).isQuestion;
}

// ---- the user's names in raw ASR text ----

const FOLD: Record<string, string> = { ә: "а", ғ: "г", қ: "к", ң: "н", ө: "о", ұ: "у", ү: "у", һ: "х", і: "и", ё: "е" };

/**
 * Lowercases and folds Kazakh-only letters to their Russian look-alikes, so «Айгерим»
 * typed in settings matches «айгерімге» from the recognizer.
 */
export function foldForMatch(text: string): string {
  return text.toLowerCase().replace(/[әғқңөұүһіё]/g, (c) => FOLD[c] ?? c);
}

function foldedTokens(text: string): string[] {
  return foldForMatch(text).split(/[^a-zа-я0-9]+/).filter(Boolean);
}

const LATIN_DIGRAPHS: Array<[string, string]> = [
  ["sh", "ш"], ["ch", "ч"], ["zh", "ж"], ["kh", "х"], ["ts", "ц"], ["ya", "я"], ["yu", "ю"], ["yo", "е"],
];
const LATIN: Record<string, string> = {
  a: "а", b: "б", c: "к", d: "д", e: "е", f: "ф", g: "г", h: "х", i: "и", j: "дж", k: "к", l: "л", m: "м",
  n: "н", o: "о", p: "п", q: "к", r: "р", s: "с", t: "т", u: "у", v: "в", w: "в", x: "кс", y: "й", z: "з",
};

/** Naive Latin → Cyrillic spelling ("Hugo" → "хуго", "Aidos" → "айдос"): the recognizer writes Cyrillic. */
export function latinToCyrillic(word: string): string {
  const w = word.toLowerCase();
  let out = "";
  for (let i = 0; i < w.length; i++) {
    const pair = LATIN_DIGRAPHS.find(([lat]) => w.startsWith(lat, i));
    if (pair) {
      out += pair[1];
      i += pair[0].length - 1;
      continue;
    }
    const c = w[i];
    // "i" after a vowel is a glide: Aidos → айдос.
    if (c === "i" && i > 0 && "aeou".includes(w[i - 1])) out += "й";
    else out += LATIN[c] ?? c;
  }
  return out;
}

/**
 * Case endings a name can carry in speech. Kazakh (folded: ға→га, қа→ка, ның→нын, …):
 * dative, genitive, accusative, locative, ablative, instrumental, equative, plural,
 * the affectionate -жан. Russian: the endings of consonant-final names (Айдоса, Айдосу,
 * Айдосом, Айдосе) and of surnames (Ивановым, Ивановой).
 */
const NAME_SUFFIXES = new Set([
  "га", "ге", "ка", "ке", "на", "не", "нын", "нин", "дын", "дин", "тын", "тин", "ны", "ни", "ды", "ди", "ты", "ти", "н",
  "да", "де", "та", "те", "нда", "нде", "дан", "ден", "тан", "тен", "нан", "нен", "мен", "бен", "пен", "ша", "ше",
  "дай", "дей", "тай", "тей", "лар", "лер", "дар", "дер", "тар", "тер", "жан",
  "а", "у", "ом", "е", "ым", "ой", "ы", "ю", "ем", "ей", "я",
]);
/** Russian endings replacing a final -а/-я: Дана → Даны, Дане, Дану, Даной; Мария → Марии, Марию. */
const A_STEM_SUFFIXES = new Set(["ы", "и", "е", "у", "ю", "ой", "ей", "ою", "ею"]);
/** Russian endings replacing a final -й/-ь: Андрей → Андрея, Андрею, Андреем; Игорь → Игоря. */
const SOFT_STEM_SUFFIXES = new Set(["я", "ю", "ем", "е", "и"]);

/** Does one recognized token spell the (folded) name word, possibly with a case ending? */
export function tokenMatchesName(token: string, word: string): boolean {
  if (token === word) return true;
  if (word.length < 3) return false;
  if (token.startsWith(word) && NAME_SUFFIXES.has(token.slice(word.length))) return true;
  const last = word.at(-1)!;
  const stem = word.slice(0, -1);
  if ((last === "а" || last === "я") && token.startsWith(stem) && A_STEM_SUFFIXES.has(token.slice(stem.length))) return true;
  if ((last === "й" || last === "ь") && token.startsWith(stem) && SOFT_STEM_SUFFIXES.has(token.slice(stem.length))) return true;
  return false;
}

export type NameHit = {
  /** The configured name that matched. */
  name: string;
  /** Token index of the match. */
  index: number;
  /** In calling position: first words of the utterance (after fillers) or its last word. */
  vocative: boolean;
};

/** Spelling variants to look for: the folded name, plus a Cyrillic spelling of a Latin one. */
function nameVariants(name: string): string[][] {
  const words = foldedTokens(name);
  if (!words.length) return [];
  const variants = [words];
  if (words.some((w) => /[a-z]/.test(w))) variants.push(words.map((w) => (/[a-z]/.test(w) ? latinToCyrillic(w) : w)));
  return variants;
}

/**
 * Finds the first of `names` in raw ASR text, case-insensitively and allowing Kazakh and
 * Russian case endings: «хугоға», «хуго а вы», «айдосқа», «айгерімге», «с ерланом».
 * Multi-word names must appear as consecutive words.
 */
export function findName(text: string, names: readonly string[]): NameHit | null {
  const tokens = foldedTokens(text);
  if (!tokens.length) return null;
  let start = 0;
  while (start < tokens.length - 1 && start < 3 && FILLERS.has(tokens[start])) start++;
  let best: NameHit | null = null;
  for (const name of names) {
    for (const words of nameVariants(name)) {
      for (let i = 0; i + words.length <= tokens.length; i++) {
        if (!words.every((w, k) => tokenMatchesName(tokens[i + k], w))) continue;
        const vocative = i <= start + 1 || i + words.length === tokens.length;
        if (!best || i < best.index) best = { name, index: i, vocative };
        break;
      }
    }
  }
  return best;
}

// ---- automatic hints ----

export const AUTO_HINT_MIN_WORDS = 3;
/** Addressed by name: "Хуго, а ты?" is enough. */
export const NAMED_HINT_MIN_WORDS = 2;
export const AUTO_HINT_DEBOUNCE_MS = 20_000;
/** A direct address by name is high-signal, so it waits less after the previous auto hint. */
export const NAMED_HINT_DEBOUNCE_MS = 8_000;
/** How long everybody must stay quiet after an unnamed question before it counts as possibly the user's. */
export const SILENCE_WAIT_MS = 2_500;
/** "The one just before it from the same speaker" must have ended at most this long before. */
export const SAME_SPEAKER_GAP_MS = 10_000;

export type AutoHintInput = {
  /** A final segment just added to the transcript. */
  segment: Segment;
  /** The segment is the user's own speech (label "me", or a label the user named with one of `myNames`). */
  own: boolean;
  /** The segment right before, if the same speaker said it (see `previousSameSpeaker`). */
  previous: Segment | null;
  mode: AutoHintMode;
  myNames: readonly string[];
  nowMs: number;
  /** When the last automatic hint fired, epoch ms, or null. */
  lastAutoHintAt: number | null;
  /** A hint is currently streaming or queued. */
  hintBusy: boolean;
  /** An API key is configured. */
  hasApiKey: boolean;
};

export type AutoHintDecision =
  | { action: "fire"; kind: "any" | "named"; reason: string }
  | { action: "wait"; kind: "silence"; reason: string; waitMs: number }
  | { action: "skip"; reason: string };

/**
 * Auto-hint trigger for one new final segment. Never fires on the user's own speech.
 *
 * - mode "any": a lexical question (≥ 3 words) from anyone else, ≥ 20 s after the last auto hint.
 * - mode "addressed":
 *   (a) "named": the segment, or the same speaker's previous one, contains one of the user's
 *       names, and the segment is a question or starts/ends with the name (≥ 2 words, 8 s debounce);
 *   (b) "silence": a lexical question without the name → "wait"; the caller fires it if nobody
 *       speaks for `waitMs`, and the task lets Claude decide (SKIP if not for the user).
 */
export function evaluateAutoHint(i: AutoHintInput): AutoHintDecision {
  const skip = (reason: string): AutoHintDecision => ({ action: "skip", reason });
  if (i.mode === "off") return skip("disabled");
  if (!i.hasApiKey) return skip("no-key");
  if (!i.segment.isFinal) return skip("partial");
  if (i.own) return skip("own-speech");
  const q = analyzeQuestion(i.segment.text);
  const since = i.lastAutoHintAt === null ? Infinity : i.nowMs - i.lastAutoHintAt;

  if (i.mode === "any") {
    if (q.words < AUTO_HINT_MIN_WORDS) return skip("too-short");
    if (!q.isQuestion) return skip("not-question");
    if (i.hintBusy) return skip("busy");
    if (since < AUTO_HINT_DEBOUNCE_MS) return skip("debounce");
    return { action: "fire", kind: "any", reason: q.reason ?? "question" };
  }

  const hit = findName(i.segment.text, i.myNames);
  const prevHit = !hit && i.previous ? findName(i.previous.text, i.myNames) : null;
  if ((hit || prevHit) && (q.isQuestion || hit?.vocative)) {
    if (q.words < NAMED_HINT_MIN_WORDS && !prevHit) return skip("too-short");
    if (i.hintBusy) return skip("busy");
    if (since < NAMED_HINT_DEBOUNCE_MS) return skip("debounce");
    return { action: "fire", kind: "named", reason: `name:${(hit ?? prevHit)!.name}${q.isQuestion ? `+${q.reason}` : ""}` };
  }
  if (q.words < AUTO_HINT_MIN_WORDS) return skip("too-short");
  if (!q.isQuestion) return skip(hit ? "name-not-addressed" : "not-question");
  if (since < AUTO_HINT_DEBOUNCE_MS) return skip("debounce");
  return { action: "wait", kind: "silence", reason: q.reason ?? "question", waitMs: SILENCE_WAIT_MS };
}

/**
 * The segment right before `seg` if the same speaker said it (same label; unlabelled segments
 * match by source) and it ended at most `maxGapMs` before `seg` started: one utterance the
 * recognizer split in two («хуго» … «а по срокам что скажешь»). If anyone else spoke in between,
 * the earlier name was about something else, so there is no previous segment.
 */
export function previousSameSpeaker(
  finals: readonly Segment[],
  seg: Segment,
  labelOf: (s: Segment) => string | null,
  maxGapMs = SAME_SPEAKER_GAP_MS,
): Segment | null {
  const key = (s: Segment) => labelOf(s) ?? `?${s.source}`;
  let prev: Segment | null = null;
  for (const f of finals) {
    if (f.id === seg.id || f.startMs > seg.startMs || (f.startMs === seg.startMs && f.id > seg.id)) continue;
    if (!prev || f.startMs > prev.startMs || (f.startMs === prev.startMs && f.id > prev.id)) prev = f;
  }
  return prev && key(prev) === key(seg) && seg.startMs - prev.endMs <= maxGapMs ? prev : null;
}

/**
 * Condition (b) of the "addressed" mode: an unnamed question waits for silence. Any new
 * speech (a partial or a final other than the question itself) cancels it: somebody,
 * possibly the user, is already answering.
 */
export class SilenceWatch {
  private pending: { segment: Segment; at: number } | null = null;

  arm(segment: Segment, nowMs: number): void {
    this.pending = { segment, at: nowMs };
  }

  cancel(): void {
    this.pending = null;
  }

  get armed(): Segment | null {
    return this.pending?.segment ?? null;
  }

  /** Returns true when this speech cancelled a pending question. */
  onSpeech(seg: Pick<Segment, "id" | "text">): boolean {
    if (!this.pending || seg.id === this.pending.segment.id || !seg.text.trim()) return false;
    this.pending = null;
    return true;
  }

  /** The question, once `waitMs` of silence has passed; clears it. */
  take(nowMs: number, waitMs = SILENCE_WAIT_MS): Segment | null {
    if (!this.pending || nowMs - this.pending.at < waitMs) return null;
    const seg = this.pending.segment;
    this.pending = null;
    return seg;
  }
}

export type RollingInput = {
  /** Settings.rollingSummaryMinutes; 0 = off. */
  minutes: number;
  nowMs: number;
  /** When the session started running, epoch ms. */
  sessionStartedAt: number;
  /** When the last rolling summary started, epoch ms, or null. */
  lastRunAt: number | null;
  /** Final segments added since the last rolling summary. */
  newFinals: number;
  /** A rolling summary is in flight. */
  busy: boolean;
  hasApiKey: boolean;
};

/** Rolling summary timer: every N minutes, and only if something new was said. */
export function rollingSummaryDue(i: RollingInput): boolean {
  if (i.minutes <= 0 || i.busy || !i.hasApiKey) return false;
  if (i.newFinals < 1) return false;
  const since = i.lastRunAt ?? i.sessionStartedAt;
  return i.nowMs - since >= i.minutes * 60_000;
}

/** Speaker-name suggestions run at most once per rolling-summary interval (this when that is off). */
export const DEFAULT_SUGGEST_MINUTES = 4;
/** And only if at least this many new lines arrived since the last run. */
export const SUGGEST_MIN_NEW_FINALS = 4;

export type SuggestDueInput = {
  /** Settings.rollingSummaryMinutes (0 = rolling summary off; suggestions then use the default). */
  intervalMinutes: number;
  nowMs: number;
  sessionStartedAt: number;
  lastRunAt: number | null;
  newFinals: number;
  /** Labels that have spoken, have no name and no open suggestion. */
  unnamed: number;
  busy: boolean;
  hasApiKey: boolean;
};

export function suggestionsDue(i: SuggestDueInput): boolean {
  if (i.busy || !i.hasApiKey || i.unnamed < 1 || i.newFinals < SUGGEST_MIN_NEW_FINALS) return false;
  const minutes = i.intervalMinutes > 0 ? i.intervalMinutes : DEFAULT_SUGGEST_MINUTES;
  return i.nowMs - (i.lastRunAt ?? i.sessionStartedAt) >= minutes * 60_000;
}
