import { formatTranscript } from "../llm/prompts";
import type { Namer } from "../llm/speakers";
import type { Segment } from "../types";

export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function formatTime(ts: number | string): string {
  const d = new Date(ts);
  return d.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
}

export function formatDateTime(ts: number | string): string {
  const d = new Date(ts);
  return d.toLocaleString("ru-RU", { day: "numeric", month: "long", hour: "2-digit", minute: "2-digit" });
}

export function formatDuration(startIso: string, endIso: string | null): string | null {
  if (!endIso) return null;
  const ms = new Date(endIso).getTime() - new Date(startIso).getTime();
  if (!Number.isFinite(ms) || ms < 0) return null;
  const min = Math.round(ms / 60000);
  return min < 60 ? `${min} мин` : `${Math.floor(min / 60)} ч ${min % 60} мин`;
}

export function fileSlug(title: string): string {
  const slug = title
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
  return slug || "meeting";
}

export function isoDate(ts: number | string): string {
  const d = new Date(ts);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/** Markdown export of a meeting: summary first, then the transcript. */
export function meetingMarkdown(opts: {
  title: string;
  startedAt: number | string | null;
  summary: string | null;
  rolling?: string | null;
  segments: Segment[];
  /** Resolves speaker display names (custom names, «Участник N», …). */
  speakers: Namer;
}): string {
  const parts = [`# ${opts.title || "Встреча"}`];
  if (opts.startedAt) parts.push(`_${formatDateTime(opts.startedAt)}_`);
  if (opts.summary?.trim()) parts.push(opts.summary.trim());
  else if (opts.rolling?.trim()) parts.push(`## Резюме по ходу встречи\n\n${opts.rolling.trim()}`);
  const transcript = formatTranscript(opts.segments, opts.speakers);
  if (transcript) parts.push(`## Расшифровка\n\n${transcript.split("\n").map((l) => `${l}  `).join("\n")}`);
  return `${parts.join("\n\n")}\n`;
}
