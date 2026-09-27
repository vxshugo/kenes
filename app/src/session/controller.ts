import type Anthropic from "@anthropic-ai/sdk";
import { getBackend, type Backend } from "../backend";
import { applyFormat, type MeetingFormat } from "../lib/meetingFormat";
import { createClient, describeError, isAbort, type StreamOutcome } from "../llm/client";
import { Conversation } from "../llm/conversation";
import { Copilot } from "../llm/copilot";
import { SYSTEM_PROMPT, compareSegments, formatContextBlock, formatMeetingDate } from "../llm/prompts";
import { applyRelabel, isRenamable, SpeakerDirectory } from "../llm/speakers";
import { buildSuggestRequest, parseSuggestions, SUGGEST_REQUEST, unnamedLabels, type NameSuggestion } from "../llm/suggestions";
import {
  askTask,
  explainTask,
  finalSummaryTask,
  hintTask,
  isSkipReply,
  mayBeSkip,
  pickTranslationLines,
  recapTask,
  rollingSummaryTask,
  translateTask,
  type AutoKind,
  type TaskSpec,
} from "../llm/tasks";
import {
  AUTO_HINT_DEBOUNCE_MS,
  evaluateAutoHint,
  foldForMatch,
  previousSameSpeaker,
  rollingSummaryDue,
  SilenceWatch,
  SILENCE_WAIT_MS,
  suggestionsDue,
} from "../llm/triggers";
import { DEFAULT_SETTINGS } from "../types";
import type {
  EnrollResult,
  MicMode,
  PipelineEvent,
  Segment,
  SessionState,
  Settings,
  Source,
  SpeakerChange,
  VoiceprintStatus,
} from "../types";

export type Phase = "idle" | "starting" | "loading" | "running" | "stopping" | "stopped" | "error";

export type CardKind = "hint" | "ask" | "explain" | "translate" | "recap";
export type CardStatus = "queued" | "streaming" | "done" | "error";

export type Card = {
  id: string;
  kind: CardKind;
  label: string;
  /** What triggered it: the question, the term, the user's question. */
  detail: string | null;
  auto: boolean;
  status: CardStatus;
  text: string;
  error: string | null;
  truncated: boolean;
  /** Extra line under the card, e.g. which fallback model answered. */
  note: string | null;
  createdAt: number;
  /**
   * An auto hint that may still turn out to be SKIP (the question wasn't for the user):
   * not shown until its text diverges from "SKIP", dropped silently if it is SKIP.
   */
  hidden: boolean;
};

export type SuggestionChip = NameSuggestion & { id: string };

export type SummaryDoc = {
  status: "idle" | "streaming" | "done" | "error";
  text: string;
  error: string | null;
  truncated: boolean;
  updatedAt: number | null;
  note: string | null;
};

export type Toast = { id: number; text: string; tone: "info" | "error" };

export type ControllerState = {
  ready: boolean;
  initError: string | null;
  backendKind: "tauri" | "mock";
  settings: Settings;
  apiKey: string | null;
  phase: Phase;
  statusMessage: string | null;
  modelProgress: { model: string; progress: number } | null;
  error: string | null;
  meetingId: string | null;
  title: string;
  context: string;
  /** Wall-clock start (Date.now()) for the elapsed timer. */
  startedAt: number | null;
  endedAt: number | null;
  captureSystem: boolean;
  captureMic: boolean;
  /** The session's mic mode (fixed at start; `settings.micMode` before a start). */
  micMode: MicMode;
  finals: Segment[];
  partials: Segment[];
  cards: Card[];
  rolling: SummaryDoc;
  final: SummaryDoc;
  toasts: Toast[];
  /** Custom speaker names of the current meeting, label → name. */
  speakerNames: Record<string, string>;
  /** Name suggestions from Claude for unnamed speakers. */
  suggestions: SuggestionChip[];
  suggesting: boolean;
  voiceprint: VoiceprintStatus | null;
  /** `enroll_voice` is recording. */
  enrolling: boolean;
};

export type Levels = Record<Source, number>;

const EMPTY_DOC: SummaryDoc = { status: "idle", text: "", error: null, truncated: false, updatedAt: null, note: null };

const INITIAL: ControllerState = {
  ready: false,
  initError: null,
  backendKind: "mock",
  settings: DEFAULT_SETTINGS,
  apiKey: null,
  phase: "idle",
  statusMessage: null,
  modelProgress: null,
  error: null,
  meetingId: null,
  title: "",
  context: "",
  startedAt: null,
  endedAt: null,
  captureSystem: true,
  captureMic: true,
  micMode: "me",
  finals: [],
  partials: [],
  cards: [],
  rolling: EMPTY_DOC,
  final: EMPTY_DOC,
  toasts: [],
  speakerNames: {},
  suggestions: [],
  suggesting: false,
  voiceprint: null,
  enrolling: false,
};

const ROLLING_TICK_MS = 10_000;
const STOP_GRACE_MS = 400;
const MAX_CARDS = 30;
const RELABEL_NOTE = "После итогов диаризация уточнила говорящих — «Сгенерировать заново», чтобы учесть.";

type CardOptions = { auto?: boolean; onSkip?: () => void };

export type ControllerDeps = {
  /** Defaults to Tauri inside the desktop shell, the scripted mock in a browser. */
  backend?: Backend;
  createClient?: (apiKey: string) => Anthropic;
};
function outcomeNote(out: StreamOutcome, requested: string): string | null {
  if (out.fallback || (out.model && out.model !== requested && !out.model.startsWith(requested))) {
    return `Ответила резервная модель ${out.model}`;
  }
  return null;
}

/**
 * Owns the live session: backend events, the transcript, the Claude conversation,
 * auto-hint and rolling-summary triggers, notes. React reads it through
 * `useSyncExternalStore` (see `useController`).
 */
export class SessionController {
  private state: ControllerState = INITIAL;
  private listeners = new Set<() => void>();
  private levelListeners = new Set<(l: Levels) => void>();
  private levels: Levels = { mic: 0, system: 0 };
  private backend: Backend | null = null;
  private unlisten: (() => void) | null = null;
  private copilot: Copilot | null = null;
  private clientCache: { key: string; client: Anthropic } | null = null;
  private rollingTimer: ReturnType<typeof setInterval> | null = null;
  private runningSince: number | null = null;
  private lastAutoHintAt: number | null = null;
  private lastRollingAt: number | null = null;
  private finalsSinceRolling = 0;
  private liveInFlight = 0;
  private toastSeq = 0;
  private cardSeq = 0;
  private notifyScheduled = false;
  private initPromise: Promise<void> | null = null;
  private cardSpecs = new Map<string, { kind: CardKind; spec: TaskSpec; detail: string | null; opts: CardOptions }>();
  private speakers: SpeakerDirectory | null = null;
  private readonly silence = new SilenceWatch();
  private silenceTimer: ReturnType<typeof setTimeout> | null = null;
  private lastSuggestAt: number | null = null;
  private finalsSinceSuggest = 0;
  private rejected: Array<{ label: string; name: string }> = [];
  private suggestSeq = 0;
  private relabeledAfterFinal = false;
  private settingsSaving: Promise<unknown> = Promise.resolve();
  private startPending = false;

  constructor(private readonly deps: ControllerDeps = {}) {}

  // ---- store plumbing ----

  getState = (): ControllerState => this.state;

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  subscribeLevels = (fn: (l: Levels) => void): (() => void) => {
    this.levelListeners.add(fn);
    fn(this.levels);
    return () => this.levelListeners.delete(fn);
  };

  private set(patch: Partial<ControllerState> | ((s: ControllerState) => Partial<ControllerState>)) {
    const p = typeof patch === "function" ? patch(this.state) : patch;
    this.state = { ...this.state, ...p };
    // Coalesce bursts (streaming deltas, partials) into one render per frame. rAF stalls
    // while the window is hidden or occluded, so a timer backs it up.
    if (!this.notifyScheduled) {
      this.notifyScheduled = true;
      const flush = () => {
        if (!this.notifyScheduled) return;
        this.notifyScheduled = false;
        for (const l of this.listeners) l();
      };
      if (typeof requestAnimationFrame === "function") requestAnimationFrame(flush);
      setTimeout(flush, 100);
    }
  }

  private toast(text: string, tone: Toast["tone"] = "error") {
    const id = ++this.toastSeq;
    this.set((s) => ({ toasts: [...s.toasts.slice(-3), { id, text, tone }] }));
    setTimeout(() => this.dismissToast(id), tone === "error" ? 8000 : 4000);
  }

  dismissToast(id: number) {
    this.set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
  }

  // ---- init & settings ----

  init(): Promise<void> {
    if (!this.initPromise) this.initPromise = this.doInit();
    return this.initPromise;
  }

  private async doInit() {
    try {
      const backend = this.deps.backend ?? (await getBackend());
      this.backend = backend;
      const [settings, apiKey, voiceprint] = await Promise.all([
        backend.getSettings(),
        backend.getApiKey().catch(() => null),
        backend.voiceprintStatus().catch(() => null),
      ]);
      this.unlisten = await backend.onEvent((e) => this.onEvent(e));
      this.set({ ready: true, backendKind: backend.kind, settings, apiKey: apiKey || null, voiceprint, micMode: settings.micMode });
    } catch (e) {
      this.set({ ready: true, initError: describeError(e) });
    }
  }

  dispose() {
    this.unlisten?.();
    this.unlisten = null;
    this.stopTimers();
    this.copilot?.close();
  }

  get api(): Backend {
    if (!this.backend) throw new Error("backend not ready");
    return this.backend;
  }

  async saveSettings(settings: Settings) {
    const saving = this.api.saveSettings(settings).then(() => {
      this.set((s) => ({ settings, micMode: this.isActive || s.phase === "stopping" || s.phase === "stopped" ? s.micMode : settings.micMode }));
    });
    this.settingsSaving = saving.catch(() => undefined);
    await saving;
  }

  /** The meeting format from the pre-start sheet, persisted before `start_session` reads it. */
  async setMeetingFormat(format: MeetingFormat) {
    try {
      await this.saveSettings(applyFormat(this.state.settings, format));
    } catch (e) {
      this.toast(`Не удалось сохранить формат встречи: ${describeError(e)}`);
    }
  }

  async setApiKey(key: string) {
    const k = key.trim();
    await this.api.setApiKey(k);
    this.clientCache = null;
    this.set({ apiKey: k || null });
  }

  private getClient = (): Anthropic | null => {
    const key = this.state.apiKey;
    if (!key) return null;
    if (this.clientCache?.key !== key) this.clientCache = { key, client: (this.deps.createClient ?? createClient)(key) };
    return this.clientCache.client;
  };

  // ---- session lifecycle ----

  get isActive(): boolean {
    return ["starting", "loading", "running"].includes(this.state.phase);
  }

  private get isStopping(): boolean {
    return this.state.phase === "stopping";
  }

  async start(title: string, context: string) {
    if (this.isActive || this.state.phase === "stopping" || this.startPending || this.state.enrolling) return;
    this.startPending = true;
    try {
      // A format picked right before Start must reach the backend before start_session reads it.
      await this.settingsSaving;
    } finally {
      this.startPending = false;
    }
    if (this.isActive || this.isStopping) return;
    const settings = this.state.settings;
    const cleanTitle = title.trim() || "Встреча";
    this.copilot?.close();
    const speakers = new SpeakerDirectory(settings.micMode);
    this.speakers = speakers;
    const conversation = new Conversation(
      SYSTEM_PROMPT,
      formatContextBlock({
        title: cleanTitle,
        context,
        profile: settings.profile,
        setup: { captureMic: settings.captureMic, captureSystem: settings.captureSystem, micMode: settings.micMode },
        voiceprint: !!this.state.voiceprint?.enrolled,
        myNames: settings.myNames,
        date: formatMeetingDate(new Date()),
      }),
      speakers,
    );
    this.copilot = new Copilot(conversation, {
      getClient: this.getClient,
      getConfig: () => ({
        model: this.state.settings.claudeModel,
        hintEffort: this.state.settings.hintEffort,
        summaryEffort: this.state.settings.summaryEffort,
      }),
    });
    this.cardSpecs.clear();
    this.lastAutoHintAt = null;
    this.lastRollingAt = null;
    this.finalsSinceRolling = 0;
    this.lastSuggestAt = null;
    this.finalsSinceSuggest = 0;
    this.rejected = [];
    this.relabeledAfterFinal = false;
    this.cancelSilence();
    this.liveInFlight = 0;
    this.runningSince = null;
    this.levels = { mic: 0, system: 0 };
    this.emitLevels();
    this.set({
      phase: "starting",
      statusMessage: null,
      modelProgress: null,
      error: null,
      meetingId: null,
      title: cleanTitle,
      context,
      startedAt: Date.now(),
      endedAt: null,
      captureSystem: settings.captureSystem,
      captureMic: settings.captureMic,
      micMode: settings.micMode,
      finals: [],
      partials: [],
      cards: [],
      rolling: EMPTY_DOC,
      final: EMPTY_DOC,
      speakerNames: {},
      suggestions: [],
      suggesting: false,
    });
    try {
      const { meetingId } = await this.api.startSession(cleanTitle, context);
      this.set((s) => ({ meetingId, phase: s.phase === "starting" ? "loading" : s.phase }));
      this.startTimers();
    } catch (e) {
      this.set({ phase: "error", error: `Не удалось начать запись: ${describeError(e)}` });
    }
  }

  async stop() {
    if (!this.isActive && this.state.phase !== "error") return;
    const wasError = this.state.phase === "error";
    this.set({ phase: "stopping" });
    this.stopTimers();
    try {
      await this.api.stopSession();
    } catch (e) {
      if (!wasError) this.toast(`Ошибка при остановке: ${describeError(e)}`);
    }
    await new Promise((r) => setTimeout(r, STOP_GRACE_MS));
    this.finishSession();
  }

  /** Session ended (by Stop or by the backend): wrap up and write the final summary. */
  private finishSession() {
    if (this.state.phase === "stopped" || this.state.phase === "idle") return;
    this.stopTimers();
    this.cancelSilence();
    this.levels = { mic: 0, system: 0 };
    this.emitLevels();
    this.set({ phase: "stopped", endedAt: Date.now(), partials: [], statusMessage: null, modelProgress: null });
    if (this.state.finals.length && this.state.apiKey) void this.generateFinal();
  }

  /** Back to the pre-start state (keeps the finished meeting in history). */
  reset() {
    if (this.isActive || this.state.phase === "stopping") return;
    this.copilot?.close();
    this.copilot = null;
    this.speakers = null;
    this.cardSpecs.clear();
    this.cancelSilence();
    this.set({
      phase: "idle",
      meetingId: null,
      title: "",
      context: "",
      startedAt: null,
      endedAt: null,
      finals: [],
      partials: [],
      cards: [],
      rolling: EMPTY_DOC,
      final: EMPTY_DOC,
      error: null,
      statusMessage: null,
      modelProgress: null,
      speakerNames: {},
      suggestions: [],
      suggesting: false,
      micMode: this.state.settings.micMode,
      captureMic: this.state.settings.captureMic,
      captureSystem: this.state.settings.captureSystem,
    });
  }

  private startTimers() {
    this.stopTimers();
    this.rollingTimer = setInterval(() => {
      this.maybeRolling();
      this.maybeSuggest();
    }, ROLLING_TICK_MS);
  }

  private stopTimers() {
    if (this.rollingTimer !== null) clearInterval(this.rollingTimer);
    this.rollingTimer = null;
  }

  // ---- backend events ----

  private onEvent(e: PipelineEvent) {
    switch (e.type) {
      case "segment":
        this.onSegment(e);
        break;
      case "level":
        this.levels = { ...this.levels, [e.source]: e.rms };
        this.emitLevels();
        break;
      case "status":
        this.onStatus(e.state, e.message);
        break;
      case "modelProgress":
        this.set({ modelProgress: { model: e.model, progress: e.progress } });
        break;
      case "error":
        this.toast(e.message);
        break;
      case "speakersRelabeled":
        this.onRelabel(e.changes ?? []);
        break;
    }
  }

  private emitLevels() {
    for (const l of this.levelListeners) l(this.levels);
  }

  private onStatus(state: SessionState, message: string | null) {
    const phase = this.state.phase;
    switch (state) {
      case "loading":
        if (phase === "starting" || phase === "loading") this.set({ phase: "loading", statusMessage: message });
        break;
      case "running":
        if (phase === "starting" || phase === "loading" || phase === "running") {
          if (this.runningSince === null) this.runningSince = Date.now();
          this.set({ phase: "running", statusMessage: message, modelProgress: null });
        }
        break;
      case "idle":
        if (phase === "running" || phase === "loading" || phase === "starting") this.finishSession();
        break;
      case "error":
        this.set({ phase: "error", error: message || "Ошибка распознавания", statusMessage: null });
        this.stopTimers();
        break;
    }
  }

  private onSegment(seg: Segment) {
    if (!this.copilot) return;
    const text = seg.text.trim();
    // Any new speech breaks the silence an unnamed question is waiting for.
    if (text && this.silence.onSpeech(seg)) this.clearSilenceTimer();
    if (!seg.isFinal) {
      this.set((s) => {
        const others = s.partials.filter((p) => p.id !== seg.id);
        return { partials: text ? [...others, seg].sort(compareSegments) : others };
      });
      return;
    }
    const final: Segment = { ...seg, speaker: seg.speaker ?? null };
    this.set((s) => {
      const partials = s.partials.filter((p) => p.id !== seg.id);
      if (!text || s.finals.some((f) => f.id === seg.id)) return { partials };
      const finals = [...s.finals, final];
      // Keep display order by start time; out-of-order arrivals are rare and nearby.
      for (let i = finals.length - 1; i > 0 && compareSegments(finals[i - 1], finals[i]) > 0; i--) {
        [finals[i - 1], finals[i]] = [finals[i], finals[i - 1]];
      }
      return { partials, finals };
    });
    if (!this.copilot.conversation.addFinal(final)) return;
    this.finalsSinceRolling++;
    this.finalsSinceSuggest++;
    this.considerAutoHint(final);
  }

  /** The user's own speech: "me", or a label the user named with one of their own names. */
  private isOwnLabel(label: string | null): boolean {
    if (label === "me") return true;
    const custom = this.speakers?.customName(label);
    if (!custom) return false;
    const folded = foldForMatch(custom);
    return this.state.settings.myNames.some((n) => foldForMatch(n) === folded);
  }

  private considerAutoHint(seg: Segment) {
    const speakers = this.speakers;
    if (!speakers || this.state.phase !== "running") return;
    const settings = this.state.settings;
    const label = speakers.labelOf(seg);
    const decision = evaluateAutoHint({
      segment: seg,
      own: this.isOwnLabel(label),
      previous: previousSameSpeaker(this.state.finals, seg, (s) => speakers.labelOf(s)),
      mode: settings.autoHintMode,
      myNames: settings.myNames,
      nowMs: Date.now(),
      lastAutoHintAt: this.lastAutoHintAt,
      hintBusy: this.liveInFlight > 0,
      hasApiKey: !!this.state.apiKey,
    });
    if (decision.action === "fire") {
      this.autoHint(seg, decision.kind);
    } else if (decision.action === "wait") {
      this.silence.arm(seg, Date.now());
      this.clearSilenceTimer();
      this.silenceTimer = setTimeout(() => this.onSilence(), decision.waitMs + 20);
    }
  }

  /** Condition (b): an unnamed question met silence; Claude decides whether it was for the user. */
  private onSilence() {
    this.silenceTimer = null;
    const seg = this.silence.take(Date.now(), SILENCE_WAIT_MS);
    if (!seg || this.state.phase !== "running" || this.state.settings.autoHintMode !== "addressed" || !this.state.apiKey) return;
    if (this.state.partials.length || this.liveInFlight > 0) return;
    if (this.lastAutoHintAt !== null && Date.now() - this.lastAutoHintAt < AUTO_HINT_DEBOUNCE_MS) return;
    this.autoHint(seg, "silence");
  }

  private clearSilenceTimer() {
    if (this.silenceTimer !== null) clearTimeout(this.silenceTimer);
    this.silenceTimer = null;
  }

  private cancelSilence() {
    this.silence.cancel();
    this.clearSilenceTimer();
  }

  private autoHint(seg: Segment, kind: AutoKind) {
    const speakers = this.speakers;
    if (!speakers) return;
    const settings = this.state.settings;
    const previousAt = this.lastAutoHintAt;
    const at = Date.now();
    this.lastAutoHintAt = at;
    const spec = hintTask({
      lang: settings.answerLanguage,
      speakers,
      question: seg,
      auto: kind,
      skippable: settings.autoHintMode === "addressed",
      myNames: settings.myNames,
    });
    this.runCard("hint", spec, `${speakers.nameFor(seg)}: ${seg.text}`, {
      auto: true,
      // A SKIP answer doesn't count against the debounce.
      onSkip: () => {
        if (this.lastAutoHintAt === at) this.lastAutoHintAt = previousAt;
      },
    });
  }

  // ---- speakers ----

  private onRelabel(changes: readonly SpeakerChange[]) {
    const speakers = this.speakers;
    if (!speakers || !changes.length || this.state.phase === "idle") return;
    const finals = applyRelabel(this.state.finals, changes);
    this.copilot?.conversation.relabel(changes);
    if (finals === this.state.finals) return;
    const present = new Set(finals.map((f) => speakers.labelOf(f)));
    const finalBusy = this.state.final.status === "streaming" || this.state.final.status === "done";
    if (finalBusy) this.relabeledAfterFinal = true;
    this.set((s) => ({
      finals,
      suggestions: s.suggestions.filter((x) => present.has(x.label)),
      final: s.final.status === "done" ? { ...s.final, note: RELABEL_NOTE } : s.final,
    }));
  }

  /**
   * Names a speaker (empty or the default name resets it). Applies everywhere at once; Claude
   * learns about it from a note in the next turn. Persisted with `rename_speaker`.
   */
  async renameSpeaker(label: string, name: string) {
    const speakers = this.speakers;
    if (!speakers || !isRenamable(label)) return;
    const before = speakers.customName(label);
    if (!speakers.setName(label, name)) return;
    const after = speakers.customName(label);
    this.set((s) => ({ speakerNames: speakers.snapshot(), suggestions: s.suggestions.filter((x) => x.label !== label) }));
    const meetingId = this.state.meetingId;
    if (!meetingId) return;
    try {
      await this.api.renameSpeaker(meetingId, label, after ?? "");
    } catch (e) {
      if (speakers !== this.speakers) return;
      speakers.setName(label, before);
      this.set({ speakerNames: speakers.snapshot() });
      this.toast(`Не удалось переименовать: ${describeError(e)}`);
    }
  }

  acceptSuggestion(id: string) {
    const s = this.state.suggestions.find((x) => x.id === id);
    if (s) void this.renameSpeaker(s.label, s.name);
  }

  dismissSuggestion(id: string) {
    const s = this.state.suggestions.find((x) => x.id === id);
    if (!s) return;
    this.rejected.push({ label: s.label, name: s.name });
    this.set((st) => ({ suggestions: st.suggestions.filter((x) => x.id !== id) }));
  }

  private maybeSuggest() {
    const speakers = this.speakers;
    if (!speakers || this.state.phase !== "running") return;
    const open = new Set(this.state.suggestions.map((x) => x.label));
    const due = suggestionsDue({
      intervalMinutes: this.state.settings.rollingSummaryMinutes,
      nowMs: Date.now(),
      sessionStartedAt: this.runningSince ?? this.state.startedAt ?? Date.now(),
      lastRunAt: this.lastSuggestAt,
      newFinals: this.finalsSinceSuggest,
      unnamed: unnamedLabels(this.state.finals, speakers).filter((l) => !open.has(l)).length,
      busy: this.state.suggesting,
      hasApiKey: !!this.state.apiKey,
    });
    if (due) void this.suggestNames(false);
  }

  /**
   * Asks Claude which unnamed speakers can be named from the dialogue. A separate small
   * request with structured output; the meeting conversation (and its cache) is not touched.
   */
  async suggestNames(manual = true) {
    const copilot = this.copilot;
    const speakers = this.speakers;
    if (!copilot || !speakers || this.state.suggesting) return;
    if (!this.state.apiKey) {
      if (manual) this.toast("Не задан API-ключ Claude. Добавьте его в «Настройках».");
      return;
    }
    if (!unnamedLabels(this.state.finals, speakers).length) {
      if (manual) this.toast("Все говорящие уже подписаны.", "info");
      return;
    }
    this.lastSuggestAt = Date.now();
    this.finalsSinceSuggest = 0;
    const model = this.state.settings.claudeModel;
    const input = { finals: this.state.finals, speakers, myNames: this.state.settings.myNames, rejected: [...this.rejected] };
    this.set({ suggesting: true });
    try {
      const out = await copilot.runSide(buildSuggestRequest(input), SUGGEST_REQUEST);
      if (speakers !== this.speakers) return;
      const found = parseSuggestions(out.text, input).filter((f) => !speakers.hasCustomName(f.label));
      this.set((s) => {
        const replaced = new Set(found.map((f) => f.label));
        const fresh = found.map((f) => ({ ...f, id: `sug-${++this.suggestSeq}` }));
        return { suggesting: false, suggestions: [...s.suggestions.filter((x) => !replaced.has(x.label)), ...fresh] };
      });
      if (manual && !found.length) this.toast("В разговоре пока нет подсказок, как зовут безымянных говорящих.", "info");
    } catch (err) {
      if (speakers !== this.speakers) return;
      this.set({ suggesting: false });
      if (manual && !isAbort(err)) this.toast(`Не удалось предложить имена: ${describeError(err, model)}`);
    }
  }

  // ---- voiceprint ----

  async refreshVoiceprint() {
    try {
      this.set({ voiceprint: await this.api.voiceprintStatus() });
    } catch {
      this.set({ voiceprint: null });
    }
  }

  /** Records the user's voice (no session may be running). */
  async enrollVoice(seconds: number): Promise<EnrollResult> {
    if (this.isActive || this.state.phase === "stopping" || this.state.enrolling) throw new Error("Сначала завершите встречу.");
    this.set({ enrolling: true });
    try {
      return await this.api.enrollVoice(seconds);
    } finally {
      this.set({ enrolling: false });
      await this.refreshVoiceprint();
    }
  }

  async clearVoiceprint() {
    await this.api.clearVoiceprint();
    await this.refreshVoiceprint();
  }

  // ---- live tasks ----

  private newCard(kind: Card["kind"], spec: TaskSpec, detail: string | null, auto: boolean): Card {
    return {
      hidden: !!spec.skippable,
      id: `card-${++this.cardSeq}`,
      kind,
      label: spec.label,
      detail,
      auto,
      status: "queued",
      text: "",
      error: null,
      truncated: false,
      note: null,
      createdAt: Date.now(),
    };
  }

  private patchCard(id: string, patch: Partial<Card>) {
    this.set((s) => ({ cards: s.cards.map((c) => (c.id === id ? { ...c, ...patch } : c)) }));
  }

  private runCard(kind: Card["kind"], spec: TaskSpec, detail: string | null, opts: CardOptions = {}) {
    const copilot = this.copilot;
    if (!copilot) {
      this.toast("Сначала начните встречу.", "info");
      return;
    }
    if (!this.state.apiKey) {
      this.toast("Не задан API-ключ Claude. Добавьте его в «Настройках».");
      return;
    }
    const card = this.newCard(kind, spec, detail, !!opts.auto);
    this.cardSpecs.set(card.id, { kind, spec, detail, opts });
    this.set((s) => {
      const cards = [card, ...s.cards];
      for (const dropped of cards.slice(MAX_CARDS)) this.cardSpecs.delete(dropped.id);
      return { cards: cards.slice(0, MAX_CARDS) };
    });
    this.liveInFlight++;
    const meetingId = this.state.meetingId;
    const model = this.state.settings.claudeModel;
    copilot
      .runLive(spec, {
        onStart: () => this.patchCard(card.id, { status: "streaming" }),
        // A skippable card stays hidden while its text could still be "SKIP".
        onText: (_d, text) => this.patchCard(card.id, spec.skippable && mayBeSkip(text) ? { text } : { text, hidden: false }),
      })
      .then((out) => {
        if (spec.skippable && isSkipReply(out.text)) {
          // Not for the user: no card, no note, no debounce.
          this.dropCard(card.id);
          opts.onSkip?.();
          return;
        }
        this.patchCard(card.id, { status: "done", hidden: false, text: out.text, truncated: out.truncated, note: outcomeNote(out, model) });
        if (meetingId) this.persist(meetingId, "hint", out.text, spec.trigger);
      })
      .catch((err) => {
        if (isAbort(err)) {
          if (spec.skippable) this.dropCard(card.id);
          else this.patchCard(card.id, { status: "error", error: "Отменено.", text: "" });
          return;
        }
        // A refusal's partial text is discarded, never shown as an answer.
        this.patchCard(card.id, { status: "error", hidden: false, error: describeError(err, model), text: "" });
      })
      .finally(() => {
        this.liveInFlight = Math.max(0, this.liveInFlight - 1);
      });
  }

  private dropCard(cardId: string) {
    this.cardSpecs.delete(cardId);
    this.set((s) => ({ cards: s.cards.filter((c) => c.id !== cardId) }));
  }

  /** Re-runs a failed card's task as a new card. */
  retry(cardId: string) {
    const entry = this.cardSpecs.get(cardId);
    if (!entry) return;
    this.dismissCard(cardId);
    this.runCard(entry.kind, entry.spec, entry.detail, entry.opts);
  }

  dismissCard(cardId: string) {
    this.dropCard(cardId);
  }

  /** The current meeting's speaker names, or a toast when there is no meeting yet. */
  private sessionSpeakers(): SpeakerDirectory | null {
    if (!this.speakers || !this.copilot) {
      this.toast("Сначала начните встречу.", "info");
      return null;
    }
    return this.speakers;
  }

  /** "Что ответить?" */
  hint() {
    const speakers = this.sessionSpeakers();
    if (!speakers) return;
    const settings = this.state.settings;
    const spec = hintTask({ lang: settings.answerLanguage, speakers, partials: this.state.partials, myNames: settings.myNames });
    this.runCard("hint", spec, null);
  }

  ask(question: string) {
    const q = question.trim();
    const speakers = this.sessionSpeakers();
    if (!q || !speakers) return;
    const spec = askTask({ question: q, lang: this.state.settings.answerLanguage, speakers, partials: this.state.partials });
    this.runCard("ask", spec, q);
  }

  explain(term: string) {
    const t = term.trim();
    if (!t) return;
    this.runCard("explain", explainTask({ term: t, lang: this.state.settings.answerLanguage }), t);
  }

  translate() {
    const speakers = this.sessionSpeakers();
    if (!speakers) return;
    const lines = pickTranslationLines(this.state.finals, this.state.partials);
    if (!lines.length) {
      this.toast("Пока нечего переводить.", "info");
      return;
    }
    this.runCard("translate", translateTask({ lines, speakers }), null);
  }

  recap(minutes = 5) {
    const speakers = this.sessionSpeakers();
    if (!speakers) return;
    this.runCard("recap", recapTask({ minutes, nowMs: this.sessionNowMs(), speakers }), null);
  }

  /** Session-relative "now", from the latest segment timestamps (same clock as the transcript). */
  private sessionNowMs(): number {
    const ends = [...this.state.finals, ...this.state.partials].map((s) => s.endMs);
    const fromSegments = ends.length ? Math.max(...ends) : 0;
    const fromClock = this.runningSince ? Date.now() - this.runningSince : 0;
    return Math.max(fromSegments, fromClock);
  }

  // ---- summaries ----

  private maybeRolling() {
    if (this.state.phase !== "running") return;
    const due = rollingSummaryDue({
      minutes: this.state.settings.rollingSummaryMinutes,
      nowMs: Date.now(),
      sessionStartedAt: this.runningSince ?? this.state.startedAt ?? Date.now(),
      lastRunAt: this.lastRollingAt,
      newFinals: this.finalsSinceRolling,
      busy: this.state.rolling.status === "streaming",
      hasApiKey: !!this.state.apiKey,
    });
    if (due) void this.generateRolling();
  }

  async generateRolling() {
    const copilot = this.copilot;
    const speakers = this.speakers;
    if (!copilot || !speakers || this.state.rolling.status === "streaming") return;
    if (!this.state.apiKey) {
      this.toast("Не задан API-ключ Claude. Добавьте его в «Настройках».");
      return;
    }
    this.lastRollingAt = Date.now();
    this.finalsSinceRolling = 0;
    const previous = this.state.rolling.status === "done" ? this.state.rolling.text : null;
    const meetingId = this.state.meetingId;
    const model = this.state.settings.claudeModel;
    this.set((s) => ({ rolling: { ...s.rolling, status: "streaming", error: null, text: previous ?? "" } }));
    let started = false;
    try {
      const out = await copilot.runFork(rollingSummaryTask({ previous, speakers }), {
        onText: (_d, text) => {
          started = true;
          this.set((s) => ({ rolling: { ...s.rolling, text } }));
        },
      });
      this.set({
        rolling: { status: "done", text: out.text, error: null, truncated: out.truncated, updatedAt: Date.now(), note: outcomeNote(out, model) },
      });
      if (meetingId) this.persist(meetingId, "summary", out.text, "rolling");
    } catch (err) {
      this.set((s) => ({
        rolling: {
          ...s.rolling,
          status: previous ? "done" : "error",
          text: previous ?? (started ? "" : s.rolling.text),
          error: isAbort(err) ? null : describeError(err, model),
        },
      }));
    }
  }

  async generateFinal() {
    const copilot = this.copilot;
    const speakers = this.speakers;
    if (!copilot || !speakers || this.state.final.status === "streaming") return;
    if (!this.state.apiKey) {
      this.toast("Не задан API-ключ Claude. Добавьте его в «Настройках».");
      return;
    }
    const meetingId = this.state.meetingId;
    const model = this.state.settings.claudeModel;
    const spec = finalSummaryTask({ speakers });
    this.relabeledAfterFinal = false;
    this.set({ final: { ...EMPTY_DOC, status: "streaming" } });
    try {
      const out = await copilot.runFinal(spec, {
        onText: (_d, text) => this.set((s) => ({ final: { ...s.final, text } })),
      });
      this.set({
        final: {
          status: "done",
          text: out.text,
          error: null,
          truncated: out.truncated,
          updatedAt: Date.now(),
          note: outcomeNote(out, model) ?? (this.relabeledAfterFinal ? RELABEL_NOTE : null),
        },
      });
      if (meetingId) this.persist(meetingId, "final", out.text, spec.trigger);
    } catch (err) {
      this.set({ final: { ...EMPTY_DOC, status: "error", error: isAbort(err) ? "Отменено." : describeError(err, model) } });
    }
  }

  private persist(meetingId: string, kind: "hint" | "summary" | "final", content: string, trigger: string | null) {
    this.api.saveNote(meetingId, kind, content, trigger).catch((e) => {
      this.toast(`Не удалось сохранить заметку: ${describeError(e)}`);
    });
  }
}
