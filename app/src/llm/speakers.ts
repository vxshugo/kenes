import type { MicMode, Segment, SpeakerChange } from "../types";

/**
 * Speaker labels from the Rust diarizer (docs/CONTRACT.md), set on final segments only:
 * "me" (the user), "sys:N" (N-th voice in the call), "mic:N" (N-th voice on a room mic),
 * null (too short or uncertain). Names are per meeting and live on the Rust side
 * (`rename_speaker`); everything here is pure and shared by the UI and the Claude layer.
 */

export const ME = "me";

export type ParsedLabel = { kind: "me" } | { kind: "sys" | "mic"; n: number } | { kind: "other"; raw: string };

export function parseLabel(label: string | null | undefined): ParsedLabel | null {
  if (!label) return null;
  if (label === ME) return { kind: "me" };
  const m = /^(sys|mic):(\d+)$/.exec(label);
  if (m) return { kind: m[1] as "sys" | "mic", n: Number(m[2]) };
  return { kind: "other", raw: label };
}

/** «Я», «Участник N», «Зал N», «?». Unknown labels are shown as they are. */
export function defaultSpeakerName(label: string | null | undefined): string {
  const p = parseLabel(label);
  if (!p) return "?";
  switch (p.kind) {
    case "me":
      return "Я";
    case "sys":
      return `Участник ${p.n}`;
    case "mic":
      return `Зал ${p.n}`;
    default:
      return p.raw;
  }
}

/**
 * The label a segment belongs to. In `micMode: "me"` every mic segment is the user, so a
 * mic segment without a label (a partial, or an older backend) still counts as "me".
 */
export function effectiveLabel(seg: Pick<Segment, "source" | "speaker">, micMode: MicMode): string | null {
  if (seg.speaker) return seg.speaker;
  return seg.source === "mic" && micMode === "me" ? ME : null;
}

/** Labels the user can name. «Я» stays «Я» (it is how Claude knows which lines are the user's). */
export function isRenamable(label: string | null | undefined): label is string {
  return !!label && label !== ME;
}

/** me first, then sys:1…, mic:1…, then anything else alphabetically. */
export function compareLabels(a: string | null, b: string | null): number {
  const rank = (l: string | null): [number, number, string] => {
    const p = parseLabel(l);
    if (!p) return [4, 0, ""];
    if (p.kind === "me") return [0, 0, ""];
    if (p.kind === "other") return [3, 0, p.raw];
    return [p.kind === "sys" ? 1 : 2, p.n, ""];
  };
  const [ra, na, sa] = rank(a);
  const [rb, nb, sb] = rank(b);
  return ra - rb || na - nb || (sa < sb ? -1 : sa > sb ? 1 : 0);
}

/** Size of the categorical speaker palette (`--spk-1` … `--spk-12` in App.css). */
export const PALETTE_SIZE = 12;

/**
 * CSS class with a stable color per label: the same label gets the same color live and in
 * history. sys:N and mic:N start half a palette apart so a hybrid meeting's first voices differ.
 */
export function speakerClass(label: string | null | undefined): string {
  const p = parseLabel(label);
  if (!p) return "spk-none";
  if (p.kind === "me") return "spk-me";
  if (p.kind === "other") {
    let h = 0;
    for (const ch of p.raw) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return `spk-${(h % PALETTE_SIZE) + 1}`;
  }
  const offset = p.kind === "mic" ? PALETTE_SIZE / 2 : 0;
  return `spk-${((p.n - 1 + offset) % PALETTE_SIZE + PALETTE_SIZE) % PALETTE_SIZE + 1}`;
}

export const MAX_SPEAKER_NAME = 40;

/** Collapses whitespace and caps the length; "" means "no custom name". */
export function cleanSpeakerName(name: string): string {
  return name.replace(/\s+/g, " ").trim().slice(0, MAX_SPEAKER_NAME).trim();
}

export type SpeakerNames = Readonly<Record<string, string>>;

/** Resolves display names for segments; the transcript formatter only needs this. */
export interface Namer {
  nameFor(seg: Pick<Segment, "source" | "speaker">): string;
}

/**
 * The names of one meeting's speakers. Custom names override the defaults; a name equal to
 * the default (or empty) resets it.
 */
export class SpeakerDirectory implements Namer {
  private readonly names = new Map<string, string>();

  constructor(
    readonly micMode: MicMode,
    initial: SpeakerNames = {},
  ) {
    for (const [label, name] of Object.entries(initial)) this.setName(label, name);
  }

  labelOf(seg: Pick<Segment, "source" | "speaker">): string | null {
    return effectiveLabel(seg, this.micMode);
  }

  customName(label: string | null | undefined): string | null {
    return label ? (this.names.get(label) ?? null) : null;
  }

  nameOf(label: string | null | undefined): string {
    return this.customName(label) ?? defaultSpeakerName(label);
  }

  nameFor(seg: Pick<Segment, "source" | "speaker">): string {
    return this.nameOf(this.labelOf(seg));
  }

  /** Sets or resets (null / "" / the default name) a label's name. Returns true if it changed. */
  setName(label: string, name: string | null): boolean {
    if (!isRenamable(label)) return false;
    const clean = cleanSpeakerName(name ?? "");
    const before = this.names.get(label) ?? null;
    if (!clean || clean === defaultSpeakerName(label)) this.names.delete(label);
    else this.names.set(label, clean);
    return (this.names.get(label) ?? null) !== before;
  }

  /** Custom names, keys in label order (deterministic for prompts). */
  snapshot(): Record<string, string> {
    const out: Record<string, string> = {};
    for (const label of [...this.names.keys()].sort(compareLabels)) out[label] = this.names.get(label)!;
    return out;
  }

  hasCustomName(label: string): boolean {
    return this.names.has(label);
  }
}

export type SpeakerStat = {
  /** null collects segments nobody could be attributed to. */
  label: string | null;
  name: string;
  custom: boolean;
  talkMs: number;
  /** Contiguous runs of this speaker in transcript order (one turn may span several segments). */
  turns: number;
  segments: number;
  firstMs: number;
};

type Directory = Pick<SpeakerDirectory, "labelOf" | "nameOf" | "customName">;

/**
 * Talk time and turns per speaker from final segments, sorted by talk time (then label).
 * `extraLabels` adds speakers with no segments (e.g. named in `get_meeting.speakers`).
 */
export function speakerStats(finals: readonly Segment[], dir: Directory, extraLabels: readonly string[] = []): SpeakerStat[] {
  const byLabel = new Map<string | null, SpeakerStat>();
  const ordered = [...finals].filter((s) => s.text.trim()).sort((a, b) => a.startMs - b.startMs || (a.id < b.id ? -1 : 1));
  let prev: string | null | undefined;
  for (const seg of ordered) {
    const label = dir.labelOf(seg);
    let st = byLabel.get(label);
    if (!st) {
      st = { label, name: dir.nameOf(label), custom: !!dir.customName(label), talkMs: 0, turns: 0, segments: 0, firstMs: seg.startMs };
      byLabel.set(label, st);
    }
    st.talkMs += Math.max(0, seg.endMs - seg.startMs);
    st.segments++;
    if (prev !== label) st.turns++;
    prev = label;
  }
  for (const label of extraLabels) {
    if (!byLabel.has(label)) {
      byLabel.set(label, { label, name: dir.nameOf(label), custom: !!dir.customName(label), talkMs: 0, turns: 0, segments: 0, firstMs: Infinity });
    }
  }
  return [...byLabel.values()].sort(
    (a, b) => Number(a.label === null) - Number(b.label === null) || b.talkMs - a.talkMs || compareLabels(a.label, b.label),
  );
}

/** Applies `speakersRelabeled` to a segment list; returns the same array when nothing changed. */
export function applyRelabel<T extends Segment>(segments: T[], changes: readonly SpeakerChange[]): T[] {
  if (!changes.length) return segments;
  const map = new Map(changes.map((c) => [c.segmentId, c.speaker ?? null]));
  let changed = false;
  const out = segments.map((s) => {
    if (!map.has(s.id)) return s;
    const speaker = map.get(s.id)!;
    if (speaker === s.speaker) return s;
    changed = true;
    return { ...s, speaker };
  });
  return changed ? out : segments;
}

/** A stored meeting doesn't say how the mic was used; any "mic:N" label means a room. */
export function inferMicMode(segments: readonly Pick<Segment, "speaker">[]): MicMode {
  return segments.some((s) => s.speaker?.startsWith("mic:")) ? "room" : "me";
}

/** «45 с», «3 мин 05 с», «1 ч 02 мин». */
export function formatTalkTime(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000));
  if (total < 60) return `${total} с`;
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h) return `${h} ч ${String(m).padStart(2, "0")} мин`;
  return `${m} мин ${String(s).padStart(2, "0")} с`;
}

/** «1 реплика», «3 реплики», «11 реплик». */
export function pluralTurns(n: number): string {
  const mod10 = n % 10;
  const mod100 = n % 100;
  const word = mod10 === 1 && mod100 !== 11 ? "реплика" : mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14) ? "реплики" : "реплик";
  return `${n} ${word}`;
}
