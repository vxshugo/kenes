import type {
  DeviceInfo,
  EnrollResult,
  HotkeyAction,
  HotkeyConfig,
  HotkeyStatus,
  LiveSession,
  PlatformInfo,
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
  /** The running session, if any (`SessionManager::live()`), so a reloaded UI can reattach. */
  sessionStatus(): Promise<LiveSession | null>;
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
  /** Applies the global-shortcut settings; resolves when the system answered (the portal may ask first). */
  configureHotkeys(config: HotkeyConfig): Promise<HotkeyStatus>;
  platformInfo(): Promise<PlatformInfo>;

  /** Subscribes to `kenes://event`. Resolves to an unsubscribe function. */
  onEvent(handler: (event: PipelineEvent) => void): Promise<() => void>;
  /** Subscribes to `kenes://hotkey` (a system-wide shortcut fired). */
  onHotkey(handler: (action: HotkeyAction) => void): Promise<() => void>;
  /** Subscribes to `kenes://hotkey-status` (bindings changed, e.g. in the system settings). */
  onHotkeyStatus(handler: (status: HotkeyStatus) => void): Promise<() => void>;
}
