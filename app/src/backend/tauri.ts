import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { normalizeSettings } from "../types";
import type {
  DeviceInfo,
  EnrollResult,
  Meeting,
  MeetingSummary,
  ModelInfo,
  NoteKind,
  PipelineEvent,
  Segment,
  Settings,
  Speaker,
  VoiceprintStatus,
} from "../types";
import type { Backend } from "./types";

export const EVENT_NAME = "kenes://event";

export class TauriBackend implements Backend {
  readonly kind = "tauri" as const;

  listDevices() {
    return invoke<DeviceInfo[]>("list_devices");
  }
  listModels() {
    return invoke<ModelInfo[]>("list_models");
  }
  startSession(title: string, context: string) {
    return invoke<{ meetingId: string }>("start_session", { title, context });
  }
  stopSession() {
    return invoke<void>("stop_session");
  }
  async getSettings(): Promise<Settings> {
    return normalizeSettings(await invoke<unknown>("get_settings"));
  }
  saveSettings(settings: Settings) {
    return invoke<void>("save_settings", { settings });
  }
  getApiKey() {
    return invoke<string | null>("get_api_key");
  }
  setApiKey(key: string) {
    return invoke<void>("set_api_key", { key });
  }
  listMeetings() {
    return invoke<MeetingSummary[]>("list_meetings");
  }
  async getMeeting(id: string): Promise<Meeting> {
    return normalizeMeeting(await invoke<Meeting>("get_meeting", { id }));
  }
  saveNote(meetingId: string, kind: NoteKind, content: string, trigger: string | null) {
    return invoke<string>("save_note", { meetingId, kind, content, trigger });
  }
  deleteMeeting(id: string) {
    return invoke<void>("delete_meeting", { id });
  }
  renameSpeaker(meetingId: string, label: string, name: string) {
    return invoke<void>("rename_speaker", { meetingId, label, name });
  }
  enrollVoice(seconds: number) {
    return invoke<EnrollResult>("enroll_voice", { seconds });
  }
  voiceprintStatus() {
    return invoke<VoiceprintStatus>("voiceprint_status");
  }
  clearVoiceprint() {
    return invoke<void>("clear_voiceprint");
  }
  async onEvent(handler: (event: PipelineEvent) => void) {
    return listen<PipelineEvent>(EVENT_NAME, (e) => handler(e.payload));
  }
}

/** Tolerates an older backend: no `speakers`, segments without a `speaker` key. */
export function normalizeMeeting(m: Meeting): Meeting {
  const segments = (m.segments ?? []).map((s: Segment) => (s.speaker === undefined ? { ...s, speaker: null } : s));
  const speakers: Speaker[] = Array.isArray(m.speakers) ? m.speakers : [];
  return { ...m, segments, notes: m.notes ?? [], speakers };
}
