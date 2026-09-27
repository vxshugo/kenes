import Anthropic from "@anthropic-ai/sdk";
import type { RequestPayload } from "./conversation";

type BetaMessage = Anthropic.Beta.Messages.BetaMessage;
type StreamParams = Parameters<Anthropic["beta"]["messages"]["stream"]>[0];
type Effort = "low" | "medium" | "high" | "xhigh";

/** Beta header for the `fallbacks: "default"` scalar form (the array form uses a different one). */
export const FALLBACK_BETA = "server-side-fallback-2026-07-01";

/**
 * Client for the user's own key. This is a local desktop app talking straight to
 * api.anthropic.com from the webview, hence `dangerouslyAllowBrowser`.
 */
export function createClient(apiKey: string): Anthropic {
  return new Anthropic({ apiKey, dangerouslyAllowBrowser: true, maxRetries: 2 });
}

export type ModelCaps = {
  /** `thinking: {type: "adaptive"}` is accepted. */
  adaptiveThinking: boolean;
  /** `output_config.effort` is accepted. */
  effort: boolean;
  /** Effort level `xhigh` exists (otherwise it is lowered to `high`). */
  xhigh: boolean;
  /** Server-side `fallbacks: "default"` is offered for this model. */
  serverFallbacks: boolean;
  /** Structured outputs (`output_config.format` with a JSON schema) are accepted. */
  structuredOutputs: boolean;
};

/**
 * What request features a model id supports, from the model catalog. Unknown ids are
 * assumed to be current-generation models (adaptive thinking + effort) but don't get
 * the fallbacks beta.
 */
export function modelCaps(model: string): ModelCaps {
  const m = model.trim().toLowerCase();
  const parsed = /^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d+))?/.exec(m);
  if (!parsed) {
    if (/^claude-(3|2|instant)/.test(m)) {
      return { adaptiveThinking: false, effort: false, xhigh: false, serverFallbacks: false, structuredOutputs: false };
    }
    return { adaptiveThinking: true, effort: true, xhigh: true, serverFallbacks: false, structuredOutputs: true };
  }
  const family = parsed[1];
  const major = Number(parsed[2]);
  const minorRaw = parsed[3] ? Number(parsed[3]) : 0;
  // "claude-sonnet-4-20250514": a date, not a minor version.
  const minor = minorRaw >= 100 ? 0 : minorRaw;
  const v = major + minor / 10;
  switch (family) {
    case "fable":
    case "mythos":
      return { adaptiveThinking: true, effort: true, xhigh: true, serverFallbacks: v >= 5.1, structuredOutputs: true };
    case "opus":
      return { adaptiveThinking: v >= 4.6, effort: v >= 4.5, xhigh: v >= 4.7, serverFallbacks: v >= 5, structuredOutputs: v >= 4.1 };
    case "sonnet":
      return { adaptiveThinking: v >= 4.6, effort: v >= 4.6, xhigh: v >= 5, serverFallbacks: false, structuredOutputs: v >= 4.5 };
    default: // haiku
      return { adaptiveThinking: v >= 5, effort: v >= 5, xhigh: false, serverFallbacks: false, structuredOutputs: v >= 4.5 };
  }
}

/** Models that answered 400 to the fallbacks beta in this app run; don't send it to them again. */
const fallbacksRejected = new Set<string>();
/** Models that answered 400 to `output_config.format`; later requests ask for JSON in the prompt only. */
const formatRejected = new Set<string>();

export type JsonSchemaFormat = { type: "json_schema"; schema: Record<string, unknown> };

export type RequestOptions = {
  model: string;
  effort: Effort;
  maxTokens: number;
  /** Set false to skip the server-side fallback opt-in (e.g. after it was rejected). */
  fallbacks?: boolean;
  /** Per-request transport settings: a live hint is useless after a long wait, a final summary isn't. */
  timeoutMs?: number;
  maxRetries?: number;
  /** Structured output: the reply's text is JSON matching this schema (models that support it). */
  format?: JsonSchemaFormat;
};

/** Builds `client.beta.messages.stream()` params for the given payload and model. */
export function buildParams(payload: RequestPayload, opts: RequestOptions): StreamParams {
  const caps = modelCaps(opts.model);
  const params: StreamParams = {
    model: opts.model,
    max_tokens: opts.maxTokens,
    system: payload.system,
    messages: payload.messages,
  };
  // Adaptive thinking stays on at every effort; effort is the cost/latency lever.
  if (caps.adaptiveThinking) params.thinking = { type: "adaptive" };
  if (caps.effort) params.output_config = { effort: opts.effort === "xhigh" && !caps.xhigh ? "high" : opts.effort };
  if (opts.format && caps.structuredOutputs && !formatRejected.has(opts.model)) {
    params.output_config = { ...params.output_config, format: opts.format };
  }
  const useFallbacks = (opts.fallbacks ?? true) && caps.serverFallbacks && !fallbacksRejected.has(opts.model);
  if (useFallbacks) {
    params.fallbacks = "default";
    params.betas = [FALLBACK_BETA];
  }
  return params;
}

export class RefusalError extends Error {
  constructor(
    readonly category: string | null,
    readonly explanation: string | null,
  ) {
    super("refusal");
    this.name = "RefusalError";
  }
}

export class MissingKeyError extends Error {
  constructor() {
    super("missing api key");
    this.name = "MissingKeyError";
  }
}

export type StreamOutcome = {
  text: string;
  /** Hit `max_tokens` (or the context window): text is usable but cut off. */
  truncated: boolean;
  /** Model that produced the reply (differs from the requested one after a fallback). */
  model: string;
  /** A server-side fallback model served (part of) this reply. */
  fallback: boolean;
  usage: {
    input: number;
    cacheRead: number;
    cacheWrite: number;
    output: number;
  };
};

export type StreamHandlers = {
  /** Called with each text delta and the accumulated visible text. */
  onText?: (delta: string, text: string) => void;
};

/**
 * Streams one request, rendering text deltas as they arrive, and validates the final
 * message before its content is trusted: `refusal` throws (partial text is discarded),
 * `max_tokens` / `model_context_window_exceeded` mark the text as truncated.
 */
export async function streamMessage(
  client: Anthropic,
  payload: RequestPayload,
  opts: RequestOptions,
  handlers: StreamHandlers = {},
  signal?: AbortSignal,
): Promise<StreamOutcome> {
  const params = buildParams(payload, opts);
  let text = "";
  try {
    const stream = client.beta.messages.stream(params, {
      signal,
      ...(opts.timeoutMs ? { timeout: opts.timeoutMs } : {}),
      ...(opts.maxRetries !== undefined ? { maxRetries: opts.maxRetries } : {}),
    });
    stream.on("text", (delta) => {
      text += delta;
      handlers.onText?.(delta, text);
    });
    const message = await stream.finalMessage();
    return interpret(message);
  } catch (err) {
    // The fallbacks beta is new; if this model/account rejects it, retry once without it.
    if (err instanceof Anthropic.BadRequestError && params.fallbacks && !text) {
      fallbacksRejected.add(opts.model);
      return streamMessage(client, payload, { ...opts, fallbacks: false }, handlers, signal);
    }
    // Same for structured outputs: the caller parses JSON from plain text as a fallback.
    if (err instanceof Anthropic.BadRequestError && params.output_config?.format && !text) {
      formatRejected.add(opts.model);
      return streamMessage(client, payload, { ...opts, format: undefined }, handlers, signal);
    }
    throw err;
  }
}

function interpret(message: BetaMessage): StreamOutcome {
  if (message.stop_reason === "refusal") {
    throw new RefusalError(message.stop_details?.category ?? null, message.stop_details?.explanation ?? null);
  }
  const text = message.content
    .map((block) => (block.type === "text" ? block.text : ""))
    .join("")
    .trim();
  const fallback = (message.usage.iterations ?? []).some((it) => it.type === "fallback_message");
  return {
    text,
    truncated: message.stop_reason === "max_tokens" || message.stop_reason === "model_context_window_exceeded",
    model: message.model,
    fallback,
    usage: {
      input: message.usage.input_tokens,
      cacheRead: message.usage.cache_read_input_tokens ?? 0,
      cacheWrite: message.usage.cache_creation_input_tokens ?? 0,
      output: message.usage.output_tokens,
    },
  };
}

export function isAbort(err: unknown): boolean {
  return err instanceof Anthropic.APIUserAbortError || (err instanceof DOMException && err.name === "AbortError");
}

/** Friendly Russian message for any error thrown by the LLM layer. */
export function describeError(err: unknown, model?: string): string {
  if (err instanceof MissingKeyError) return "Не задан API-ключ Claude. Добавьте его в «Настройках».";
  if (err instanceof RefusalError) {
    const cat = err.category ? ` (категория: ${err.category})` : "";
    return `Claude отказался отвечать на этот запрос${cat}. Попробуйте переформулировать.`;
  }
  if (isAbort(err)) return "Запрос отменён.";
  if (err instanceof Anthropic.AuthenticationError) return "Ключ API не принят (401). Проверьте ключ в «Настройках».";
  if (err instanceof Anthropic.PermissionDeniedError)
    return `Нет доступа (403): у ключа нет прав на модель${model ? ` «${model}»` : ""} или функцию.`;
  if (err instanceof Anthropic.NotFoundError)
    return `Модель${model ? ` «${model}»` : ""} не найдена (404). Проверьте название модели в «Настройках».`;
  if (err instanceof Anthropic.RateLimitError) return "Превышен лимит запросов к Claude (429). Подождите немного и повторите.";
  if (err instanceof Anthropic.BadRequestError) return `Запрос отклонён API (400): ${shortApiMessage(err)}`;
  if (err instanceof Anthropic.APIConnectionTimeoutError) return "Claude не ответил вовремя. Проверьте соединение и повторите.";
  if (err instanceof Anthropic.APIConnectionError)
    return "Нет соединения с api.anthropic.com. Проверьте интернет, VPN или прокси.";
  if (err instanceof Anthropic.InternalServerError) {
    return err.status === 529
      ? "Claude сейчас перегружен (529). Повторите через минуту."
      : `Ошибка на стороне Claude (${err.status}). Повторите позже.`;
  }
  if (err instanceof Anthropic.APIError) return `Ошибка API Claude${err.status ? ` (${err.status})` : ""}: ${shortApiMessage(err)}`;
  if (err instanceof Error) return err.message;
  return String(err);
}

function shortApiMessage(err: InstanceType<typeof Anthropic.APIError>): string {
  const body = err.error as { error?: { message?: unknown } } | undefined;
  const msg = typeof body?.error?.message === "string" ? body.error.message : err.message;
  return msg.length > 220 ? `${msg.slice(0, 219)}…` : msg;
}
