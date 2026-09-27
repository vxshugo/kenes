import type Anthropic from "@anthropic-ai/sdk";
import type { HintEffort, SummaryEffort } from "../types";
import { contextWindow, PREPARE_FRACTION, ROLLOVER_FRACTION, SEED_TRANSCRIPT_FRACTION } from "./budget";
import type { Conversation, RequestPayload, RolloverInfo } from "./conversation";
import { MissingKeyError, streamMessage, type RequestOptions, type StreamHandlers, type StreamOutcome } from "./client";
import { budgetFor, type TaskRoute, type TaskSpec } from "./tasks";
import { promptTokens } from "./usage";

export type CopilotConfig = {
  model: string;
  hintEffort: HintEffort;
  summaryEffort: SummaryEffort;
};

export type CopilotDeps = {
  /** Current client, or null when no API key is set. */
  getClient: () => Anthropic | null;
  /** Read at request time so settings changes apply to the next request. */
  getConfig: () => CopilotConfig;
  /** The latest rolling summary: what a rollover carries over from the old log. */
  getSummary?: () => string | null;
  /** Every response's usage, for the cost display. */
  onUsage?: (out: StreamOutcome, route: TaskRoute | "side") => void;
  /** The log is nearing the rollover point; a fresh rolling summary would make a better seed. */
  onContextPressure?: () => void;
  onRollover?: (info: RolloverInfo) => void;
};

export class EmptyReplyError extends Error {
  constructor() {
    super("Claude вернул пустой ответ.");
    this.name = "EmptyReplyError";
  }
}

/**
 * Runs tasks against one meeting's conversation.
 * - Live tasks go through a FIFO queue: the log is append-only, so turns can't overlap.
 * - The rolling summary is a fork of the log's cached prefix and runs alongside.
 * - The final summary is a separate full-transcript request at summaryEffort.
 * - Before a live turn or a fork, the request size is estimated (calibrated on the last
 *   response's `usage`); past `ROLLOVER_FRACTION` of the model's context window the log rolls
 *   over to a new one seeded with the rolling summary and the recent transcript.
 */
export class Copilot {
  private queue: Promise<unknown> = Promise.resolve();
  private readonly controllers = new Set<AbortController>();
  private closed = false;

  constructor(
    readonly conversation: Conversation,
    private readonly deps: CopilotDeps,
  ) {}

  private client(): Anthropic {
    const c = this.deps.getClient();
    if (!c) throw new MissingKeyError();
    return c;
  }

  /**
   * Keeps the next request inside the context window: asks for a fresh summary when the log
   * nears the limit and rolls over past it. Never while a live turn waits for its reply.
   */
  private checkContext(route: "live" | "fork", taskText: string): void {
    const cfg = this.deps.getConfig();
    const window = contextWindow(cfg.model);
    const conv = this.conversation;
    const estimate = () =>
      (route === "live" ? conv.estimateTurn(taskText) : conv.estimateFork(taskText)) + budgetFor(route, cfg.hintEffort, cfg.summaryEffort).maxTokens;
    const next = estimate();
    if (next > window * PREPARE_FRACTION) {
      const epoch = conv.epoch;
      this.deps.onContextPressure?.();
      // The hook starts a rolling-summary fork, whose own check may already have rolled the log
      // over; `next` describes the old log then, so don't roll over a second time.
      if (conv.epoch !== epoch) return;
    }
    if (next <= window * ROLLOVER_FRACTION || conv.awaitingReply) return;
    const info = conv.rollover({
      summary: this.deps.getSummary?.() ?? null,
      maxTranscriptChars: Math.floor((window * SEED_TRANSCRIPT_FRACTION) / conv.tokensPerChar()),
    });
    this.deps.onRollover?.(info);
  }

  private async send(route: TaskRoute, payload: RequestPayload, handlers: StreamHandlers): Promise<StreamOutcome> {
    const cfg = this.deps.getConfig();
    const budget = budgetFor(route, cfg.hintEffort, cfg.summaryEffort);
    const ctrl = new AbortController();
    this.controllers.add(ctrl);
    try {
      const out = await streamMessage(
        this.client(),
        payload,
        {
          model: cfg.model,
          effort: budget.effort,
          maxTokens: budget.maxTokens,
          timeoutMs: route === "live" ? 90_000 : route === "fork" ? 180_000 : 600_000,
          maxRetries: route === "live" ? 1 : 2,
        },
        handlers,
        ctrl.signal,
      );
      if (route !== "final") this.conversation.observe(payload, promptTokens(out.usage));
      this.deps.onUsage?.(out, route);
      return out;
    } finally {
      this.controllers.delete(ctrl);
    }
  }

  /** Queues a live task; `onStart` fires when it leaves the queue and its request begins. */
  runLive(spec: TaskSpec, handlers: StreamHandlers & { onStart?: () => void } = {}): Promise<StreamOutcome> {
    const run = async (): Promise<StreamOutcome> => {
      if (this.closed) throw new DOMException("closed", "AbortError");
      this.client(); // fail fast without touching the log when there is no key
      handlers.onStart?.();
      this.checkContext("live", spec.text);
      const payload = this.conversation.beginTurn(spec.text);
      try {
        const out = await this.send("live", payload, handlers);
        if (!out.text) throw new EmptyReplyError();
        this.conversation.commit(out.text);
        return out;
      } catch (err) {
        // Failed request (error, refusal, abort): the only edit the log allows.
        this.conversation.rollback();
        throw err;
      }
    };
    const p = this.queue.then(run, run);
    this.queue = p.catch(() => undefined);
    return p;
  }

  /** Rolling summary: reads the cached conversation prefix, leaves the log untouched. */
  async runFork(spec: TaskSpec, handlers: StreamHandlers = {}): Promise<StreamOutcome> {
    this.checkContext("fork", spec.text);
    const out = await this.send("fork", this.conversation.fork(spec.text), handlers);
    if (!out.text) throw new EmptyReplyError();
    return out;
  }

  /** Final summary over the full transcript (summary + recent lines if it wouldn't fit). */
  async runFinal(spec: TaskSpec, handlers: StreamHandlers = {}): Promise<StreamOutcome> {
    const cfg = this.deps.getConfig();
    const tokens = contextWindow(cfg.model) * ROLLOVER_FRACTION - budgetFor("final", cfg.hintEffort, cfg.summaryEffort).maxTokens;
    const payload = this.conversation.finalRequest(spec.text, {
      maxChars: Math.floor(tokens / this.conversation.tokensPerChar()),
      summary: this.deps.getSummary?.() ?? null,
    });
    const out = await this.send("final", payload, handlers);
    if (!out.text) throw new EmptyReplyError();
    return out;
  }

  /**
   * A small side request that never touches the conversation (speaker-name suggestions).
   * Runs alongside live turns; aborted by `close()` like everything else.
   */
  async runSide(payload: RequestPayload, opts: Omit<RequestOptions, "model">): Promise<StreamOutcome> {
    if (this.closed) throw new DOMException("closed", "AbortError");
    const client = this.client();
    const ctrl = new AbortController();
    this.controllers.add(ctrl);
    try {
      const out = await streamMessage(client, payload, { ...opts, model: this.deps.getConfig().model }, {}, ctrl.signal);
      this.deps.onUsage?.(out, "side");
      return out;
    } finally {
      this.controllers.delete(ctrl);
    }
  }

  /** Aborts everything in flight; queued live tasks fail with an abort. */
  close(): void {
    this.closed = true;
    for (const c of this.controllers) c.abort();
    this.controllers.clear();
  }
}
