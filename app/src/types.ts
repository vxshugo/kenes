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

/** `session_status`: the running session, for a UI that (re)loads mid-meeting. */
export type LiveSession = { meetingId: string; state: SessionState };

/** What a system-wide shortcut does (`kenes://hotkey` payload). */
export type HotkeyAction = "hint" | "recap" | "toggle";
/** Accelerators like `CommandOrControl+Alt+Enter` (modifiers, then a `KeyboardEvent.code`). */
export type HotkeyBindings = Record<HotkeyAction, string>;
/** `configure_hotkeys` argument. */
export type HotkeyConfig = { enabled: boolean } & HotkeyBindings;
export type HotkeyBackend = "portal" | "plugin" | "none";
export type HotkeyStatus = {
  /** portal: XDG GlobalShortcuts (Wayland, the system owns the keys); plugin: macOS / X11 key grab. */
  backend: HotkeyBackend;
  state: "off" | "pending" | "active" | "cancelled" | "unavailable" | "error";
  /** What each action is bound to, as the system describes it; null = not bound. */
  bindings: Array<{ action: HotkeyAction; trigger: string | null }>;
  message: string | null;
};
export type PlatformInfo = {
  os: string;
  /** "wayland" / "x11" on Linux, null elsewhere (and in the browser mock). */
  sessionType: string | null;
  desktop: string | null;
  gnome: boolean;
  /** The window runs under XWayland because of `gnomeAlwaysOnTop`. */
  x11Forced: boolean;
  hotkeyBackend: HotkeyBackend;
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

export const STT_BACKENDS = ["auto", "ort", "sherpa"] as const;
export type SttBackend = (typeof STT_BACKENDS)[number];

export type Settings = {
  // read by Rust
  sttModel: string;
  numThreads: number;
  captureMic: boolean;
  captureSystem: boolean;
  micMode: MicMode;
  micDevice: string | null;
  systemDevice: string | null;
  /** Echo cancellation of the call audio picked up by the mic (kenes-aec); matters without headphones. */
  echoCancellation: boolean;
  /** Recognizer runtime: "auto" (per-model default, ONNX Runtime for GigaAM), "ort" or "sherpa". Next session. */
  sttBackend: SttBackend;
  /** Read by the app shell at start-up: run under XWayland on GNOME Wayland so the window can stay on top. */
  gnomeAlwaysOnTop: boolean;
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
  /** System-wide shortcuts (`configure_hotkeys`). */
  globalHotkeys: boolean;
  hotkeys: HotkeyBindings;
};

export const DEFAULT_HOTKEYS: HotkeyBindings = {
  hint: "CommandOrControl+Alt+Enter",
  recap: "CommandOrControl+Alt+KeyK",
  toggle: "CommandOrControl+Alt+KeyP",
};

export const DEFAULT_SETTINGS: Settings = {
  sttModel: "gigaam-multilingual-ctc",
  numThreads: 4,
  captureMic: true,
  captureSystem: true,
  micMode: "me",
  micDevice: null,
  systemDevice: null,
  echoCancellation: true,
  sttBackend: "auto",
  gnomeAlwaysOnTop: true,
  claudeModel: "claude-opus-5",
  hintEffort: "low",
  summaryEffort: "high",
  autoHintMode: "any",
  myNames: [],
  rollingSummaryMinutes: 4,
  answerLanguage: "auto",
  profile: "",
  globalHotkeys: true,
  hotkeys: DEFAULT_HOTKEYS,
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

const ACCELERATOR = /^(?:(?:CommandOrControl|Control|Ctrl|Alt|Shift|Super)\+)+[A-Za-z0-9]+$/;

/** A usable accelerator: at least one modifier other than Shift, then one key. */
export function isAccelerator(v: unknown): v is string {
  if (typeof v !== "string" || !ACCELERATOR.test(v)) return false;
  const parts = v.split("+");
  return parts.slice(0, -1).some((m) => m !== "Shift");
}

function normalizeHotkeys(v: unknown): HotkeyBindings {
  const r = (v && typeof v === "object" ? v : {}) as Record<string, unknown>;
  const pick = (k: keyof HotkeyBindings) => (isAccelerator(r[k]) ? (r[k] as string) : DEFAULT_HOTKEYS[k]);
  return { hint: pick("hint"), recap: pick("recap"), toggle: pick("toggle") };
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
    echoCancellation: bool(r.echoCancellation, d.echoCancellation),
    sttBackend: oneOf(r.sttBackend, STT_BACKENDS, d.sttBackend),
    gnomeAlwaysOnTop: bool(r.gnomeAlwaysOnTop, d.gnomeAlwaysOnTop),
    claudeModel: str(r.claudeModel, d.claudeModel).trim(),
    hintEffort: oneOf(r.hintEffort, HINT_EFFORTS, d.hintEffort),
    summaryEffort: oneOf(r.summaryEffort, SUMMARY_EFFORTS, d.summaryEffort),
    autoHintMode: oneOf(r.autoHintMode, AUTO_HINT_MODES, autoHintFallback),
    myNames: parseMyNames(r.myNames),
    rollingSummaryMinutes: num(r.rollingSummaryMinutes, d.rollingSummaryMinutes, 0, 120),
    answerLanguage: oneOf(r.answerLanguage, LANGUAGES, d.answerLanguage),
    profile: typeof r.profile === "string" ? r.profile : d.profile,
    globalHotkeys: bool(r.globalHotkeys, d.globalHotkeys),
    hotkeys: normalizeHotkeys(r.hotkeys),
  } as Settings;
}
