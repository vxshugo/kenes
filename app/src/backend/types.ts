import type {
  DeviceInfo,
  EnrollResult,
  Meeting,
  MeetingSummary,
  ModelInfo,
  NoteKind,
  PipelineEvent,
  Settings,
  VoiceprintStatus,
} from "../types";

/** Everything the UI needs from the Rust side (docs/CONTRACT.md), behind one interface. */
export interface Backend {
  /** "tauri" when running inside the desktop shell, "mock" in a plain browser. */
  readonly kind: "tauri" | "mock";

  listDevices(): Promise<DeviceInfo[]>;
  listModels(): Promise<ModelInfo[]>;
  startSession(title: string, context: string): Promise<{ meetingId: string }>;
  stopSession(): Promise<void>;
  getSettings(): Promise<Settings>;
  saveSettings(settings: Settings): Promise<void>;
  getApiKey(): Promise<string | null>;
  setApiKey(key: string): Promise<void>;
  listMeetings(): Promise<MeetingSummary[]>;
  getMeeting(id: string): Promise<Meeting>;
  saveNote(meetingId: string, kind: NoteKind, content: string, trigger: string | null): Promise<string>;
  deleteMeeting(id: string): Promise<void>;
  /** Names a speaker label in one meeting; an empty name resets it. */
  renameSpeaker(meetingId: string, label: string, name: string): Promise<void>;
  /** Records the mic for `seconds` and stores the user's voiceprint. No session may be running. */
  enrollVoice(seconds: number): Promise<EnrollResult>;
  voiceprintStatus(): Promise<VoiceprintStatus>;
  clearVoiceprint(): Promise<void>;

  /** Subscribes to `kenes://event`. Resolves to an unsubscribe function. */
  onEvent(handler: (event: PipelineEvent) => void): Promise<() => void>;
}
