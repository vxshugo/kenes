/**
 * Context-window budgeting for the per-meeting conversation (see `Conversation.rollover`).
 *
 * The live log is append-only for prompt caching, so it only grows. Before a request that
 * would pass `ROLLOVER_FRACTION` of the model's window, the copilot starts a new log seeded
 * with the latest rolling summary and the recent transcript. At `PREPARE_FRACTION` it asks
 * for a fresh rolling summary first, so the seed is up to date when the rollover happens.
 *
 * Server-side compaction (`compact-2026-01-12`) was considered and not used: it isn't offered
 * for Claude Haiku 4.5 (the 200K model this guards against), it needs the full response
 * content replayed (this log stores assistant turns as text), and a client-side seed keeps
 * the names map and the rolling-summary fork under our control.
 */

/** Context windows (tokens) from the model catalog; unknown or older ids get the safe 200K. */
export function contextWindow(model: string): number {
  const m = model.trim().toLowerCase();
  const parsed = /^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d+))?/.exec(m);
  if (!parsed) return 200_000;
  const family = parsed[1];
  const major = Number(parsed[2]);
  const minorRaw = parsed[3] ? Number(parsed[3]) : 0;
  const v = major + (minorRaw >= 100 ? 0 : minorRaw) / 10;
  switch (family) {
    case "fable":
    case "mythos":
      return 1_000_000;
    case "opus":
    case "sonnet":
      return v >= 4.6 ? 1_000_000 : 200_000;
    default:
      return 200_000; // Haiku 4.5
  }
}

/** Roll over when the next request (prompt + max_tokens) would pass this share of the window. */
export const ROLLOVER_FRACTION = 0.7;
/** From this share on, ask for a fresh rolling summary so the rollover has a recent one. */
export const PREPARE_FRACTION = 0.55;
/** Share of the window the recent transcript may take in a rollover seed. */
export const SEED_TRANSCRIPT_FRACTION = 0.2;
/** Tokens per character before the first response calibrates it (raw Cyrillic ASR text, generous). */
export const DEFAULT_TOKENS_PER_CHAR = 0.5;

/** Characters of all text blocks in a request (system + messages). */
export function payloadChars(p: { system: ReadonlyArray<{ text: string }>; messages: ReadonlyArray<{ content: unknown }> }): number {
  let n = 0;
  for (const b of p.system) n += b.text.length;
  for (const m of p.messages) {
    if (typeof m.content === "string") n += m.content.length;
    else if (Array.isArray(m.content)) for (const b of m.content) n += typeof b?.text === "string" ? b.text.length : 0;
  }
  return n;
}
