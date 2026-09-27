import { parseUsage, type UsageTotals } from "../llm/usage";
import type { MicMode, Note, NoteKind, Segment, SessionState } from "../types";
import type { Card, CardKind, Phase } from "./controller";

/** How a saved note's `trigger` maps back to a card (see the task builders in `llm/tasks.ts`). */
export function cardMeta(trigger: string | null): { kind: CardKind; label: string; detail: string | null; auto: boolean } {
  const t = trigger ?? "";
  const after = (prefix: string) => t.slice(prefix.length).trim() || null;
  if (t === "manual") return { kind: "hint", label: "Что ответить?", detail: null, auto: false };
  if (t.startsWith("auto:")) return { kind: "hint", label: "Подсказка (авто)", detail: after("auto:"), auto: true };
  if (t.startsWith("ask:")) return { kind: "ask", label: "Вопрос ассистенту", detail: after("ask:"), auto: false };
  if (t.startsWith("explain:")) return { kind: "explain", label: "Объяснение", detail: after("explain:"), auto: false };
  if (t === "translate") return { kind: "translate", label: "Перевод", detail: null, auto: false };
  const recap = /^recap:\s*(\d+)m$/.exec(t);
  if (recap) return { kind: "recap", label: `Кратко: ${recap[1]} мин`, detail: null, auto: false };
  return { kind: "hint", label: "Подсказка", detail: t || null, auto: false };
}

/** Saved hint notes as finished cards, newest first (the Live tab's order). */
export function cardsFromNotes(notes: readonly Note[], max: number): Card[] {
  return notes
    .filter((n) => n.kind === "hint")
    .map((n) => ({ n, at: Date.parse(n.createdAt) }))
    .sort((a, b) => b.at - a.at)
    .slice(0, max)
    .map(({ n, at }) => ({
      ...cardMeta(n.trigger),
      id: `restored-${n.id}`,
      status: "done" as const,
      text: n.content,
      error: null,
      truncated: false,
      note: null,
      createdAt: Number.isFinite(at) ? at : Date.now(),
      hidden: false,
    }));
}

export function latestNote(notes: readonly Note[], kind: NoteKind): Note | null {
  let best: Note | null = null;
  for (const n of notes) if (n.kind === kind && (!best || n.createdAt >= best.createdAt)) best = n;
  return best;
}

/** A stored meeting with any "mic:N" label was recorded in a room. */
export function inferMicMode(segments: readonly Segment[], fallback: MicMode): MicMode {
  return segments.some((s) => s.speaker?.startsWith("mic:")) ? "room" : fallback;
}

/** The UI phase for a running session's pipeline state (`session_status`). */
export function phaseFor(state: SessionState): Phase {
  switch (state) {
    case "running":
      return "running";
    case "error":
      return "error";
    default:
      // "idle" while a session exists means it was just started and hasn't reported yet.
      return "loading";
  }
}

const USAGE_KEY = "kenes.usage.";

/** Per-meeting usage, kept in this browser so a reload doesn't reset the counters. */
export function loadUsage(meetingId: string): UsageTotals | null {
  try {
    const raw = localStorage.getItem(USAGE_KEY + meetingId);
    return raw ? parseUsage(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

export function saveUsage(meetingId: string, usage: UsageTotals): void {
  try {
    localStorage.setItem(USAGE_KEY + meetingId, JSON.stringify(usage));
  } catch {
    // storage full or blocked: the counters just don't survive a reload
  }
}
