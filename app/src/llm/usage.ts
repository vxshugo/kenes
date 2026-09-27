import type { StreamOutcome } from "./client";

/** $ per million tokens. Cache writes are 1.25× input (5-minute TTL, the only one used here). */
export type ModelPrice = { input: number; output: number; cacheRead: number };

/**
 * First-party API prices from the model table in the Claude API docs (cached 2026-06-24).
 * Cache reads are 0.1× input except where the docs list their own price.
 */
const PRICES: Array<[RegExp, ModelPrice]> = [
  [/^claude-(fable|mythos)-5-1/, { input: 10, output: 50, cacheRead: 0.25 }],
  [/^claude-(fable|mythos)-5/, { input: 10, output: 50, cacheRead: 1 }],
  [/^claude-opus-5-5/, { input: 4, output: 20, cacheRead: 0.2 }],
  [/^claude-opus-(5|4-8|4-7|4-6)(?!\d)/, { input: 5, output: 25, cacheRead: 0.5 }],
  [/^claude-sonnet-5(?!\d)/, { input: 2, output: 10, cacheRead: 0.2 }],
  [/^claude-sonnet-4-6/, { input: 3, output: 15, cacheRead: 0.3 }],
  [/^claude-haiku-4-5/, { input: 1, output: 5, cacheRead: 0.1 }],
];

const CACHE_WRITE_MULTIPLIER = 1.25;

export function priceFor(model: string): ModelPrice | null {
  const m = model.trim().toLowerCase();
  return PRICES.find(([re]) => re.test(m))?.[1] ?? null;
}

export type Usage = StreamOutcome["usage"];

/** Approximate $ for one response; null when the model's price isn't known. */
export function costOf(model: string, u: Usage): number | null {
  const p = priceFor(model);
  if (!p) return null;
  return (u.input * p.input + u.cacheWrite * p.input * CACHE_WRITE_MULTIPLIER + u.cacheRead * p.cacheRead + u.output * p.output) / 1e6;
}

/** Per-meeting totals, shown in the Summary tab. */
export type UsageTotals = {
  requests: number;
  /** Uncached input tokens (`input_tokens`). */
  input: number;
  cacheRead: number;
  cacheWrite: number;
  /** Output tokens, thinking included. */
  output: number;
  /** Sum over requests with a known price. */
  cost: number;
  /** Requests whose model has no known price (their cost isn't in `cost`). */
  unpriced: number;
  /** The live conversation was restarted from a summary to stay inside the context window. */
  rollovers: number;
  /** The largest prompt sent in the live conversation, in tokens. */
  peakPrompt: number;
};

export const EMPTY_USAGE: UsageTotals = {
  requests: 0,
  input: 0,
  cacheRead: 0,
  cacheWrite: 0,
  output: 0,
  cost: 0,
  unpriced: 0,
  rollovers: 0,
  peakPrompt: 0,
};

export function promptTokens(u: Usage): number {
  return u.input + u.cacheRead + u.cacheWrite;
}

/** Adds one response. The price follows the model that answered (a fallback may differ). */
export function addUsage(t: UsageTotals, model: string, u: Usage, live = false): UsageTotals {
  const cost = costOf(model, u);
  return {
    ...t,
    requests: t.requests + 1,
    input: t.input + u.input,
    cacheRead: t.cacheRead + u.cacheRead,
    cacheWrite: t.cacheWrite + u.cacheWrite,
    output: t.output + u.output,
    cost: t.cost + (cost ?? 0),
    unpriced: t.unpriced + (cost === null ? 1 : 0),
    peakPrompt: live ? Math.max(t.peakPrompt, promptTokens(u)) : t.peakPrompt,
  };
}

/** Share of prompt tokens served from the cache (0..1), or null before any request. */
export function cacheHitRate(t: UsageTotals): number | null {
  const prompt = t.input + t.cacheRead + t.cacheWrite;
  return prompt > 0 ? t.cacheRead / prompt : null;
}

/** 1234 → "1,2K", 1_234_567 → "1,2M" (Russian decimal comma). */
export function formatTokens(n: number): string {
  const fmt = (v: number) => v.toLocaleString("ru-RU", { maximumFractionDigits: v < 10 ? 1 : 0 });
  if (n >= 1_000_000) return `${fmt(n / 1_000_000)}M`;
  if (n >= 1_000) return `${fmt(n / 1_000)}K`;
  return String(Math.round(n));
}

export function formatCost(usd: number): string {
  if (usd <= 0) return "$0";
  if (usd < 0.01) return "<$0.01";
  return `$${usd.toFixed(usd < 10 ? 2 : 1)}`;
}

/** Validates totals read back from storage. */
export function parseUsage(raw: unknown): UsageTotals | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  const out = { ...EMPTY_USAGE };
  for (const k of Object.keys(EMPTY_USAGE) as Array<keyof UsageTotals>) {
    const v = r[k];
    if (typeof v !== "number" || !Number.isFinite(v) || v < 0) return null;
    out[k] = v;
  }
  return out;
}
