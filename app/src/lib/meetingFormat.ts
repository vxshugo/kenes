import { defaultAutoHintMode, type MicMode, type Settings } from "../types";

/** The three meeting formats offered before a start. Each is a fixed audio setup. */
export type MeetingFormat = "online" | "room" | "hybrid";

export type AudioSetup = { captureMic: boolean; captureSystem: boolean; micMode: MicMode };

export const MEETING_FORMATS: Record<MeetingFormat, AudioSetup & { label: string; hint: string }> = {
  online: {
    captureMic: true,
    captureSystem: true,
    micMode: "me",
    label: "Онлайн, я в наушниках",
    hint: "Микрофон — только я, остальные — звук звонка",
  },
  room: {
    captureMic: true,
    captureSystem: false,
    micMode: "room",
    label: "В зале / офлайн",
    hint: "Один микрофон слышит всех в комнате",
  },
  hybrid: {
    captureMic: true,
    captureSystem: true,
    micMode: "room",
    label: "Гибрид",
    hint: "Зал на микрофоне плюс удалённые участники из звонка",
  },
};

export const FORMAT_ORDER: readonly MeetingFormat[] = ["online", "room", "hybrid"];

/** Which of the three formats the settings describe, or null for a custom combination. */
export function formatOf(s: AudioSetup): MeetingFormat | null {
  for (const f of FORMAT_ORDER) {
    const p = MEETING_FORMATS[f];
    if (p.captureMic === s.captureMic && p.captureSystem === s.captureSystem && p.micMode === s.micMode) return f;
  }
  return null;
}

/**
 * Settings for a chosen format. The auto-hint mode follows the mic mode's default
 * ("addressed" in a room, "any" otherwise) unless the user picked something else.
 */
export function applyFormat(s: Settings, f: MeetingFormat): Settings {
  const p = MEETING_FORMATS[f];
  const followsDefault = s.autoHintMode === defaultAutoHintMode(s.micMode);
  return {
    ...s,
    captureMic: p.captureMic,
    captureSystem: p.captureSystem,
    micMode: p.micMode,
    autoHintMode: followsDefault ? defaultAutoHintMode(p.micMode) : s.autoHintMode,
  };
}
