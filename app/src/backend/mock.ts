import { DEFAULT_SETTINGS, normalizeSettings } from "../types";
import type {
  DeviceInfo,
  EnrollResult,
  Meeting,
  MeetingSummary,
  ModelInfo,
  Note,
  NoteKind,
  PipelineEvent,
  Segment,
  Settings,
  Source,
  Speaker,
  SpeakerChange,
  VoiceprintStatus,
} from "../types";
import type { Backend } from "./types";
import { MOCK_PEOPLE, MOCK_SCRIPT, type Person, type ScriptLine } from "./mockScript";

const LS_SETTINGS = "kenes.mock.settings";
const LS_MEETINGS = "kenes.mock.meetings";
const LS_API_KEY = "kenes.mock.apiKey";
const LS_VOICEPRINT = "kenes.mock.voiceprint";

/** Mock-only defaults: the scripted meeting addresses the user as «Хуго». */
export const MOCK_DEFAULT_SETTINGS: Settings = { ...DEFAULT_SETTINGS, myNames: ["Хуго"] };

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeJson(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // storage full or blocked: the mock just forgets
  }
}

function newId(prefix: string) {
  const rnd =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;
  return `${prefix}-${rnd}`;
}

/** Replay speed; `?mockSpeed=3` in the URL makes the scripted meeting 3x faster. */
function urlSpeed(): number {
  try {
    const v = Number(new URLSearchParams(globalThis.location?.search ?? "").get("mockSpeed"));
    return Number.isFinite(v) && v > 0 ? v : 1;
  } catch {
    return 1;
  }
}

const DEVICES: DeviceInfo[] = [
  { id: "mock-mic-builtin", name: "Встроенный микрофон", kind: "input", isDefault: true },
  { id: "mock-mic-usb", name: "USB-спикерфон", kind: "input", isDefault: false },
  { id: "mock-monitor", name: "Динамики (монитор)", kind: "monitor", isDefault: true },
];

const MODELS: ModelInfo[] = [
  { id: "gigaam-multilingual-ctc", name: "GigaAM multilingual CTC", languages: ["ru", "kk"], sizeMb: 230, downloaded: true },
  { id: "whisper-large-v3-turbo", name: "Whisper large-v3 turbo", languages: ["ru", "kk", "en"], sizeMb: 1600, downloaded: false },
];

/** What the mock stores per meeting: the contract's Meeting plus the speaker names. */
type StoredMeeting = Omit<Meeting, "speakers"> & { names?: Record<string, string> };

/** Speakers as `get_meeting` returns them: every label seen or named, by talk time. */
export function speakersOf(segments: readonly Segment[], names: Record<string, string> = {}): Speaker[] {
  const map = new Map<string, Speaker>();
  for (const s of segments) {
    if (!s.speaker) continue;
    const sp = map.get(s.speaker) ?? { label: s.speaker, name: null, segmentCount: 0, talkMs: 0 };
    sp.segmentCount++;
    sp.talkMs += Math.max(0, s.endMs - s.startMs);
    map.set(s.speaker, sp);
  }
  for (const [label, name] of Object.entries(names)) {
    const sp = map.get(label) ?? { label, name: null, segmentCount: 0, talkMs: 0 };
    sp.name = name;
    map.set(label, sp);
  }
  return [...map.values()].sort((a, b) => b.talkMs - a.talkMs || a.label.localeCompare(b.label));
}

/**
 * Online speaker labels, like the Rust diarizer would assign them: per prefix, numbered by
 * first appearance, never reused. Unsure lines get null, stray lines a spurious new label;
 * `recluster()` resolves both at the end of the meeting.
 */
export class MockDiarizer {
  private readonly labels = new Map<string, string>();
  private readonly counters = { sys: 0, mic: 0 };
  /** segment id → the label re-clustering will give it. */
  private readonly corrections = new Map<string, string>();

  constructor(
    private readonly micMode: Settings["micMode"],
    private readonly voiceprint: boolean,
  ) {}

  private labelFor(person: Person, source: Source): string {
    if (person.me && (this.micMode === "me" || this.voiceprint)) return "me";
    const key = `${person.id}@${source}`;
    let label = this.labels.get(key);
    if (!label) {
      const prefix = source === "system" ? "sys" : "mic";
      label = `${prefix}:${++this.counters[prefix]}`;
      this.labels.set(key, label);
    }
    return label;
  }

  assign(segId: string, person: Person, source: Source, line: ScriptLine): string | null {
    // The user's own mic in "me" mode is always "me", even for a short "да".
    if (person.me && this.micMode === "me") return "me";
    const known = this.labels.has(`${person.id}@${source}`) || (person.me && this.voiceprint);
    if (line.unsure && known) {
      this.corrections.set(segId, this.labelFor(person, source));
      return null;
    }
    if (line.stray && known) {
      const prefix = source === "system" ? "sys" : "mic";
      this.corrections.set(segId, this.labelFor(person, source));
      return `${prefix}:${++this.counters[prefix]}`;
    }
    return this.labelFor(person, source);
  }

  /** The `speakersRelabeled` changes for segments that were actually emitted. */
  recluster(emitted: ReadonlySet<string>): SpeakerChange[] {
    return [...this.corrections].filter(([id]) => emitted.has(id)).map(([segmentId, speaker]) => ({ segmentId, speaker }));
  }
}

export type MockOptions = {
  /** Replay speed multiplier (default: `?mockSpeed=` or 1). */
  speed?: number;
  script?: ScriptLine[];
  people?: Person[];
};

type Run = { meetingId: string; cancelled: boolean; timers: ReturnType<typeof setInterval>[]; startedAt: number; diarizer: MockDiarizer; emitted: Set<string> };

/**
 * Browser stand-in for the Rust side: replays a scripted meeting with growing partials,
 * speaker labels, level meters and a fake model-loading phase; re-labels a few segments on
 * stop; keeps meetings, notes, speaker names, settings and the voiceprint in localStorage.
 */
export class MockBackend implements Backend {
  readonly kind = "mock" as const;
  private handlers = new Set<(e: PipelineEvent) => void>();
  private run: Run | null = null;
  private enrolling = false;
  private speaking: Record<Source, boolean> = { mic: false, system: false };
  private open: Partial<Record<Source, Segment>> = {};
  /** Who is speaking the open utterance, for its label when it is finalized. */
  private openLine: Partial<Record<Source, { person: Person; line: ScriptLine }>> = {};
  private counters: Record<Source, number> = { mic: 0, system: 0 };
  private readonly speed: number;
  private readonly script: ScriptLine[];
  private readonly people: Map<string, Person>;

  constructor(opts: MockOptions = {}) {
    this.speed = opts.speed ?? urlSpeed();
    this.script = opts.script ?? MOCK_SCRIPT;
    this.people = new Map((opts.people ?? MOCK_PEOPLE).map((p) => [p.id, p]));
  }

  private emit(e: PipelineEvent) {
    for (const h of this.handlers) h(e);
  }

  private wait(ms: number) {
    return delay(ms / this.speed);
  }

  async onEvent(handler: (event: PipelineEvent) => void) {
    this.handlers.add(handler);
    return () => {
      this.handlers.delete(handler);
    };
  }

  async listDevices() {
    await this.wait(120);
    return DEVICES;
  }

  async listModels() {
    await this.wait(80);
    return MODELS;
  }

  async getSettings(): Promise<Settings> {
    const stored = readJson<Record<string, unknown>>(LS_SETTINGS, {});
    return normalizeSettings({ ...MOCK_DEFAULT_SETTINGS, ...stored });
  }

  async saveSettings(settings: Settings) {
    if (settings.micMode !== "me" && settings.micMode !== "room") throw new Error(`invalid settings: micMode ${String(settings.micMode)}`);
    writeJson(LS_SETTINGS, settings);
  }

  async getApiKey() {
    try {
      const stored = localStorage.getItem(LS_API_KEY);
      if (stored) return stored;
    } catch {
      // ignore
    }
    const env = import.meta.env?.VITE_ANTHROPIC_API_KEY as string | undefined;
    return env && env.trim() ? env.trim() : null;
  }

  async setApiKey(key: string) {
    try {
      if (key) localStorage.setItem(LS_API_KEY, key);
      else localStorage.removeItem(LS_API_KEY);
    } catch {
      // ignore
    }
  }

  private meetings(): StoredMeeting[] {
    return readJson<StoredMeeting[]>(LS_MEETINGS, []);
  }

  private saveMeetings(list: StoredMeeting[]) {
    writeJson(LS_MEETINGS, list);
  }

  private updateMeeting(id: string, fn: (m: StoredMeeting) => void) {
    const list = this.meetings();
    const m = list.find((x) => x.id === id);
    if (!m) return;
    fn(m);
    this.saveMeetings(list);
  }

  async listMeetings(): Promise<MeetingSummary[]> {
    return this.meetings()
      .map(({ id, title, startedAt, endedAt }) => ({ id, title, startedAt, endedAt }))
      .sort((a, b) => b.startedAt.localeCompare(a.startedAt));
  }

  async getMeeting(id: string): Promise<Meeting> {
    const m = this.meetings().find((x) => x.id === id);
    if (!m) throw new Error(`встреча ${id} не найдена`);
    const { names, ...rest } = m;
    const segments = rest.segments.map((s) => ({ ...s, speaker: s.speaker ?? null }));
    return { ...rest, segments, speakers: speakersOf(segments, names ?? {}) };
  }

  async saveNote(meetingId: string, kind: NoteKind, content: string, trigger: string | null) {
    const note: Note = { id: newId("note"), meetingId, kind, content, trigger, createdAt: new Date().toISOString() };
    this.updateMeeting(meetingId, (m) => m.notes.push(note));
    return note.id;
  }

  async deleteMeeting(id: string) {
    if (this.run?.meetingId === id) throw new Error("нельзя удалить идущую встречу");
    this.saveMeetings(this.meetings().filter((m) => m.id !== id));
  }

  async renameSpeaker(meetingId: string, label: string, name: string) {
    if (!this.meetings().some((m) => m.id === meetingId)) throw new Error(`встреча ${meetingId} не найдена`);
    const clean = name.trim();
    this.updateMeeting(meetingId, (m) => {
      const names = { ...(m.names ?? {}) };
      if (clean) names[label] = clean;
      else delete names[label];
      m.names = names;
    });
  }

  async voiceprintStatus(): Promise<VoiceprintStatus> {
    const vp = readJson<{ createdAt: string } | null>(LS_VOICEPRINT, null);
    return vp ? { enrolled: true, createdAt: vp.createdAt } : { enrolled: false, createdAt: null };
  }

  async enrollVoice(seconds: number): Promise<EnrollResult> {
    if (this.run) throw new Error("Нельзя записывать образец голоса во время встречи.");
    if (this.enrolling) throw new Error("Запись образца уже идёт.");
    this.enrolling = true;
    const levels = setInterval(() => this.emit({ type: "level", source: "mic", rms: 0.03 + Math.random() * 0.15 }), 100);
    try {
      await this.wait(Math.max(1, seconds) * 1000);
    } finally {
      clearInterval(levels);
      this.emit({ type: "level", source: "mic", rms: 0 });
      this.enrolling = false;
    }
    writeJson(LS_VOICEPRINT, { createdAt: new Date().toISOString() });
    return { speechMs: Math.round(seconds * 1000 * (0.78 + Math.random() * 0.1)) };
  }

  async clearVoiceprint() {
    try {
      localStorage.removeItem(LS_VOICEPRINT);
    } catch {
      // ignore
    }
  }

  async startSession(title: string, context: string) {
    if (this.run) throw new Error("сессия уже идёт");
    if (this.enrolling) throw new Error("идёт запись образца голоса");
    const settings = await this.getSettings();
    const voiceprint = (await this.voiceprintStatus()).enrolled;
    const meeting: StoredMeeting = {
      id: newId("meeting"),
      title: title.trim() || "Встреча",
      startedAt: new Date().toISOString(),
      endedAt: null,
      context,
      segments: [],
      notes: [],
      names: {},
    };
    this.saveMeetings([...this.meetings(), meeting]);
    this.counters = { mic: 0, system: 0 };
    this.open = {};
    this.openLine = {};
    this.speaking = { mic: false, system: false };
    const run: Run = {
      meetingId: meeting.id,
      cancelled: false,
      timers: [],
      startedAt: 0,
      diarizer: new MockDiarizer(settings.micMode, voiceprint),
      emitted: new Set(),
    };
    this.run = run;
    void this.play(run, settings);
    return { meetingId: meeting.id };
  }

  async stopSession() {
    const run = this.run;
    if (!run) return;
    run.cancelled = true;
    run.timers.forEach((t) => clearInterval(t));
    // Like the Rust side: close whatever utterance is still open…
    for (const src of ["mic", "system"] as Source[]) {
      const seg = this.open[src];
      const who = this.openLine[src];
      if (seg) this.finalize(run, { ...seg, speaker: who ? run.diarizer.assign(seg.id, who.person, src, who.line) : null });
    }
    // …then re-cluster the whole meeting and report the labels that changed.
    const changes = run.diarizer.recluster(run.emitted);
    if (changes.length) {
      const byId = new Map(changes.map((c) => [c.segmentId, c.speaker]));
      this.updateMeeting(run.meetingId, (m) => {
        m.segments = m.segments.map((s) => (byId.has(s.id) ? { ...s, speaker: byId.get(s.id) ?? null } : s));
      });
      this.emit({ type: "speakersRelabeled", changes });
    }
    this.updateMeeting(run.meetingId, (m) => {
      m.endedAt = new Date().toISOString();
    });
    this.run = null;
    await delay(150);
    this.emit({ type: "status", state: "idle", message: null });
  }

  private finalize(run: Run, seg: Segment) {
    const final: Segment = { ...seg, isFinal: true, endMs: Math.max(seg.endMs, this.now(run)) };
    this.open[seg.source] = undefined;
    this.openLine[seg.source] = undefined;
    this.speaking[seg.source] = false;
    run.emitted.add(final.id);
    this.emit({ type: "segment", ...final });
    this.updateMeeting(run.meetingId, (m) => m.segments.push(final));
  }

  private now(run: Run) {
    return run.startedAt ? Math.round((performance.now() - run.startedAt) * this.speed) : 0;
  }

  /** Where a line is heard in this meeting format, or null if that source isn't captured. */
  private sourceFor(person: Person, settings: Settings): Source | null {
    let source: Source;
    if (person.me) source = "mic";
    else if (!settings.captureSystem) source = "mic";
    else if (settings.micMode === "me") source = "system";
    else source = person.site === "room" ? "mic" : "system";
    if (source === "mic" && !settings.captureMic) return null;
    if (source === "system" && !settings.captureSystem) return null;
    return source;
  }

  private async play(run: Run, settings: Settings) {
    // Fake model loading phase.
    this.emit({ type: "status", state: "loading", message: "Загрузка модели распознавания…" });
    for (let i = 0; i <= 10; i++) {
      if (run.cancelled) return;
      this.emit({ type: "modelProgress", model: settings.sttModel, progress: i / 10 });
      await this.wait(220);
    }
    if (run.cancelled) return;
    run.startedAt = performance.now();
    this.emit({ type: "status", state: "running", message: null });

    const levels = setInterval(() => {
      for (const src of ["mic", "system"] as Source[]) {
        const active = this.speaking[src];
        const enabled = src === "mic" ? settings.captureMic : settings.captureSystem;
        if (!enabled) continue;
        const rms = active ? 0.04 + Math.random() * 0.16 : 0.001 + Math.random() * 0.004;
        this.emit({ type: "level", source: src, rms });
      }
    }, 100);
    run.timers.push(levels);

    for (const line of this.script) {
      if (run.cancelled) return;
      await this.wait(line.pause);
      if (run.cancelled) return;
      const person = this.people.get(line.who);
      if (!person) continue;
      const source = this.sourceFor(person, settings);
      if (!source) continue;
      const id = `${source}-${++this.counters[source]}`;
      const words = line.text.split(" ");
      const startMs = this.now(run);
      this.speaking[source] = true;
      this.openLine[source] = { person, line };
      let shown = 0;
      while (shown < words.length) {
        if (run.cancelled) return;
        shown = Math.min(words.length, shown + (Math.random() < 0.3 ? 2 : 1));
        // Labels are set on final segments only.
        const seg: Segment = { id, source, speaker: null, startMs, endMs: this.now(run), text: words.slice(0, shown).join(" "), isFinal: false };
        this.open[source] = seg;
        this.emit({ type: "segment", ...seg });
        await this.wait(260 + Math.random() * 160);
      }
      await this.wait(350);
      if (run.cancelled) return;
      const seg = this.open[source];
      if (seg) this.finalize(run, { ...seg, text: line.text, speaker: run.diarizer.assign(id, person, source, line) });
    }
  }
}

function delay(ms: number) {
  return new Promise<void>((r) => setTimeout(r, ms));
}
