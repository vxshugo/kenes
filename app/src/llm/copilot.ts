import type Anthropic from "@anthropic-ai/sdk";
import type { HintEffort, SummaryEffort } from "../types";
import type { Conversation, RequestPayload } from "./conversation";
import { MissingKeyError, streamMessage, type RequestOptions, type StreamHandlers, type StreamOutcome } from "./client";
import { budgetFor, type TaskRoute, type TaskSpec } from "./tasks";

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

  private async send(route: TaskRoute, payload: RequestPayload, handlers: StreamHandlers): Promise<StreamOutcome> {
    const cfg = this.deps.getConfig();
    const budget = budgetFor(route, cfg.hintEffort, cfg.summaryEffort);
    const ctrl = new AbortController();
    this.controllers.add(ctrl);
    try {
      return await streamMessage(
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
    const out = await this.send("fork", this.conversation.fork(spec.text), handlers);
    if (!out.text) throw new EmptyReplyError();
    return out;
  }

  /** Final summary over the full transcript. */
  async runFinal(spec: TaskSpec, handlers: StreamHandlers = {}): Promise<StreamOutcome> {
    const out = await this.send("final", this.conversation.finalRequest(spec.text), handlers);
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
      return await streamMessage(client, payload, { ...opts, model: this.deps.getConfig().model }, {}, ctrl.signal);
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
