// Wire types shared with the Rust side. Source of truth: docs/CONTRACT.md.

export type Source = "mic" | "system";

export type Segment = {
  id: string;
  source: Source;
  speaker: string | null;
  startMs: number;
  endMs: number;
  text: string;
  isFinal: boolean;
};

export type SessionState = "idle" | "loading" | "running" | "error";

/** One segment whose speaker label changed after the end-of-meeting re-clustering. */
export type SpeakerChange = { segmentId: string; speaker: string | null };

export type PipelineEvent =
  | ({ type: "segment" } & Segment)
  | { type: "level"; source: Source; rms: number }
  | { type: "status"; state: SessionState; message: string | null }
  | { type: "modelProgress"; model: string; progress: number }
  | { type: "error"; message: string }
  | { type: "speakersRelabeled"; changes: SpeakerChange[] };

export type DeviceInfo = {
  id: string;
  name: string;
  kind: "input" | "monitor";
  isDefault: boolean;
};

export type ModelInfo = {
  id: string;
  name: string;
  languages: string[];
  sizeMb: number;
  downloaded: boolean;
};

export type MeetingSummary = {
  id: string;
  title: string;
  startedAt: string;
  endedAt: string | null;
};

export type NoteKind = "hint" | "summary" | "final";

export type Note = {
  id: string;
  meetingId: string;
  kind: NoteKind;
  content: string;
  trigger: string | null;
  createdAt: string;
};

/** A speaker label seen in a meeting, named or not (`get_meeting`). */
export type Speaker = {
  label: string;
  name: string | null;
  segmentCount: number;
  talkMs: number;
};

export type Meeting = MeetingSummary & {
  context: string;
  segments: Segment[];
  notes: Note[];
  speakers: Speaker[];
};

export type VoiceprintStatus = { enrolled: boolean; createdAt: string | null };
export type EnrollResult = { speechMs: number };

export type HintEffort = "low" | "medium" | "high";
export type SummaryEffort = "low" | "medium" | "high" | "xhigh";
export type AnswerLanguage = "auto" | "ru" | "kk";
/** Who the microphone hears: only the user (headset) or the whole room. */
export type MicMode = "me" | "room";
/**
 * When hints fire on their own: "addressed" (the user was named, or a question met silence;
 * Claude may answer SKIP), "any" (any question from someone else), "off".
 */
export type AutoHintMode = "addressed" | "any" | "off";

export type Settings = {
  // read by Rust
  sttModel: string;
  numThreads: number;
  captureMic: boolean;
  captureSystem: boolean;
  micMode: MicMode;
  micDevice: string | null;
  systemDevice: string | null;
  // UI only
  claudeModel: string;
  hintEffort: HintEffort;
  summaryEffort: SummaryEffort;
  /** Replaces the old boolean `autoHints` (migrated in `normalizeSettings`). */
  autoHintMode: AutoHintMode;
  /** How people address the user, e.g. ["Хуго", "Hugo"]. */
  myNames: string[];
  rollingSummaryMinutes: number;
  answerLanguage: AnswerLanguage;
  profile: string;
};

export const DEFAULT_SETTINGS: Settings = {
  sttModel: "gigaam-multilingual-ctc",
  numThreads: 4,
  captureMic: true,
  captureSystem: true,
  micMode: "me",
  micDevice: null,
  systemDevice: null,
  claudeModel: "claude-opus-5",
  hintEffort: "low",
  summaryEffort: "high",
  autoHintMode: "any",
  myNames: [],
  rollingSummaryMinutes: 4,
  answerLanguage: "auto",
  profile: "",
};

const HINT_EFFORTS: readonly HintEffort[] = ["low", "medium", "high"];
const SUMMARY_EFFORTS: readonly SummaryEffort[] = ["low", "medium", "high", "xhigh"];
const LANGUAGES: readonly AnswerLanguage[] = ["auto", "ru", "kk"];
const MIC_MODES: readonly MicMode[] = ["me", "room"];
const AUTO_HINT_MODES: readonly AutoHintMode[] = ["addressed", "any", "off"];

/** Default auto-hint mode for a mic mode: in a room only questions addressed to the user. */
export function defaultAutoHintMode(micMode: MicMode): AutoHintMode {
  return micMode === "room" ? "addressed" : "any";
}

export const MAX_MY_NAMES = 10;
const MAX_NAME_LENGTH = 40;

/** `myNames` from an array or a comma/semicolon-separated string: trimmed, deduplicated, capped. */
export function parseMyNames(v: unknown): string[] {
  const raw = Array.isArray(v) ? v : typeof v === "string" ? v.split(/[,;\n]/) : [];
  const out: string[] = [];
  const seen = new Set<string>();
  for (const item of raw) {
    if (typeof item !== "string") continue;
    const name = item.replace(/\s+/g, " ").trim().slice(0, MAX_NAME_LENGTH);
    const key = name.toLowerCase();
    if (!name || seen.has(key)) continue;
    seen.add(key);
    out.push(name);
    if (out.length >= MAX_MY_NAMES) break;
  }
  return out;
}

/** Fills gaps and drops garbage in whatever the backend returned (it may be `{}` on first run). */
export function normalizeSettings(raw: unknown): Settings {
  const r = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const d = DEFAULT_SETTINGS;
  const str = (v: unknown, fb: string) => (typeof v === "string" && v.trim() ? v : fb);
  const strOrNull = (v: unknown) => (typeof v === "string" && v ? v : null);
  const bool = (v: unknown, fb: boolean) => (typeof v === "boolean" ? v : fb);
  const num = (v: unknown, fb: number, min: number, max: number) =>
    typeof v === "number" && Number.isFinite(v) ? Math.min(max, Math.max(min, v)) : fb;
  const oneOf = <T extends string>(v: unknown, list: readonly T[], fb: T): T =>
    list.includes(v as T) ? (v as T) : fb;
  // The boolean `autoHints` is replaced by `autoHintMode`; `false` migrates to "off".
  const { autoHints: legacyAutoHints, ...rest } = r;
  const micMode = oneOf(r.micMode, MIC_MODES, d.micMode);
  const autoHintFallback: AutoHintMode = legacyAutoHints === false ? "off" : defaultAutoHintMode(micMode);
  return {
    // Keep unknown keys the Rust side may own; they round-trip through save_settings.
    ...(rest as object),
    sttModel: str(r.sttModel, d.sttModel),
    numThreads: Math.round(num(r.numThreads, d.numThreads, 1, 32)),
    captureMic: bool(r.captureMic, d.captureMic),
    captureSystem: bool(r.captureSystem, d.captureSystem),
    micMode,
    micDevice: strOrNull(r.micDevice),
    systemDevice: strOrNull(r.systemDevice),
    claudeModel: str(r.claudeModel, d.claudeModel).trim(),
    hintEffort: oneOf(r.hintEffort, HINT_EFFORTS, d.hintEffort),
    summaryEffort: oneOf(r.summaryEffort, SUMMARY_EFFORTS, d.summaryEffort),
    autoHintMode: oneOf(r.autoHintMode, AUTO_HINT_MODES, autoHintFallback),
    myNames: parseMyNames(r.myNames),
    rollingSummaryMinutes: num(r.rollingSummaryMinutes, d.rollingSummaryMinutes, 0, 120),
    answerLanguage: oneOf(r.answerLanguage, LANGUAGES, d.answerLanguage),
    profile: typeof r.profile === "string" ? r.profile : d.profile,
  } as Settings;
}
