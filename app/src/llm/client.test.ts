import Anthropic from "@anthropic-ai/sdk";
import { describe, expect, it } from "vitest";
import {
  FALLBACK_BETA,
  MissingKeyError,
  RefusalError,
  buildParams,
  describeError,
  modelCaps,
  streamMessage,
} from "./client";
import { Conversation, type RequestPayload } from "./conversation";
import { Copilot } from "./copilot";
import { SYSTEM_PROMPT } from "./prompts";
import { SpeakerDirectory } from "./speakers";
import { hintTask } from "./tasks";

const SPEAKERS = new SpeakerDirectory("me");

const payload: RequestPayload = {
  system: [{ type: "text", text: "sys", cache_control: { type: "ephemeral" } }],
  messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
};

describe("modelCaps", () => {
  it("knows the current models", () => {
    expect(modelCaps("claude-opus-5")).toEqual({ adaptiveThinking: true, effort: true, xhigh: true, serverFallbacks: true, structuredOutputs: true });
    expect(modelCaps("claude-opus-5-5").serverFallbacks).toBe(true);
    expect(modelCaps("claude-sonnet-5")).toEqual({ adaptiveThinking: true, effort: true, xhigh: true, serverFallbacks: false, structuredOutputs: true });
    expect(modelCaps("claude-haiku-4-5")).toEqual({ adaptiveThinking: false, effort: false, xhigh: false, serverFallbacks: false, structuredOutputs: true });
    expect(modelCaps("claude-haiku-4-5-20251001").effort).toBe(false);
  });
  it("handles older and dated ids", () => {
    expect(modelCaps("claude-opus-4-6")).toMatchObject({ adaptiveThinking: true, xhigh: false });
    expect(modelCaps("claude-opus-4-8")).toMatchObject({ adaptiveThinking: true, xhigh: true, serverFallbacks: false });
    expect(modelCaps("claude-sonnet-4-20250514")).toMatchObject({ adaptiveThinking: false, effort: false });
    expect(modelCaps("claude-fable-5-1").serverFallbacks).toBe(true);
  });
  it("treats unknown ids as current-generation without the fallback beta", () => {
    expect(modelCaps("claude-something-new")).toEqual({ adaptiveThinking: true, effort: true, xhigh: true, serverFallbacks: false, structuredOutputs: true });
    expect(modelCaps("claude-3-5-sonnet-20241022").structuredOutputs).toBe(false);
  });
});

describe("buildParams", () => {
  it("Claude Opus 5: adaptive thinking, effort, default server-side fallbacks with their beta", () => {
    const p = buildParams(payload, { model: "claude-opus-5", effort: "low", maxTokens: 8000 });
    expect(p).toMatchObject({
      model: "claude-opus-5",
      max_tokens: 8000,
      thinking: { type: "adaptive" },
      output_config: { effort: "low" },
      fallbacks: "default",
      betas: [FALLBACK_BETA],
    });
    expect(FALLBACK_BETA).toBe("server-side-fallback-2026-07-01");
    expect(p.system).toBe(payload.system);
    expect(p.messages).toBe(payload.messages);
    expect("temperature" in p).toBe(false);
  });
  it("drops what a model doesn't support", () => {
    const haiku = buildParams(payload, { model: "claude-haiku-4-5", effort: "high", maxTokens: 8000 });
    expect(haiku.thinking).toBeUndefined();
    expect(haiku.output_config).toBeUndefined();
    expect(haiku.fallbacks).toBeUndefined();
    expect(haiku.betas).toBeUndefined();
    const sonnet46 = buildParams(payload, { model: "claude-sonnet-4-6", effort: "xhigh", maxTokens: 8000 });
    expect(sonnet46.output_config).toEqual({ effort: "high" });
  });
  it("can opt out of fallbacks", () => {
    const p = buildParams(payload, { model: "claude-opus-5", effort: "low", maxTokens: 1, fallbacks: false });
    expect(p.fallbacks).toBeUndefined();
    expect(p.betas).toBeUndefined();
  });
});

type FakeMessage = {
  content: Array<{ type: string; text?: string }>;
  stop_reason: string;
  stop_details?: { category: string | null; explanation: string | null } | null;
  model?: string;
  usage?: Record<string, unknown>;
};

/** Minimal stand-in for `client.beta.messages.stream()`: emits text deltas, then resolves. */
function fakeClient(replies: Array<FakeMessage | Error>, deltas: string[][] = []) {
  const calls: unknown[] = [];
  const client = {
    beta: {
      messages: {
        stream(params: unknown) {
          const i = calls.length;
          calls.push(params);
          const reply = replies[i];
          const handlers: Array<(d: string) => void> = [];
          return {
            on(event: string, fn: (d: string) => void) {
              if (event === "text") handlers.push(fn);
              return this;
            },
            async finalMessage() {
              await Promise.resolve();
              if (reply instanceof Error) throw reply;
              for (const d of deltas[i] ?? []) handlers.forEach((h) => h(d));
              return {
                model: "claude-opus-5",
                stop_details: null,
                ...reply,
                usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 100, cache_creation_input_tokens: 0, ...reply.usage },
              };
            },
          };
        },
      },
    },
  };
  return { client: client as unknown as Anthropic, calls };
}

const OPTS = { model: "claude-opus-5", effort: "low" as const, maxTokens: 8000 };

describe("streamMessage", () => {
  it("streams deltas and returns the final text", async () => {
    const { client } = fakeClient([{ content: [{ type: "text", text: "Скажите: в пятницу." }], stop_reason: "end_turn" }], [["Скажите: ", "в пятницу."]]);
    const seen: string[] = [];
    const out = await streamMessage(client, payload, OPTS, { onText: (_d, text) => seen.push(text) });
    expect(seen).toEqual(["Скажите: ", "Скажите: в пятницу."]);
    expect(out).toMatchObject({ text: "Скажите: в пятницу.", truncated: false, fallback: false });
    expect(out.usage.cacheRead).toBe(100);
  });

  it("throws on refusal before reading content", async () => {
    const { client } = fakeClient([
      { content: [{ type: "text", text: "частичный" }], stop_reason: "refusal", stop_details: { category: "cyber", explanation: null } },
    ]);
    await expect(streamMessage(client, payload, OPTS)).rejects.toBeInstanceOf(RefusalError);
  });

  it("marks max_tokens as truncated", async () => {
    const { client } = fakeClient([{ content: [{ type: "thinking" }, { type: "text", text: "обре" }], stop_reason: "max_tokens" }]);
    const out = await streamMessage(client, payload, OPTS);
    expect(out).toMatchObject({ text: "обре", truncated: true });
  });

  it("reports a server-side fallback", async () => {
    const { client } = fakeClient([
      {
        content: [{ type: "fallback" }, { type: "text", text: "ok" }],
        stop_reason: "end_turn",
        model: "claude-opus-4-8",
        usage: { iterations: [{ type: "message" }, { type: "fallback_message" }] },
      },
    ]);
    const out = await streamMessage(client, payload, OPTS);
    expect(out).toMatchObject({ text: "ok", fallback: true, model: "claude-opus-4-8" });
  });

  it("retries once without the fallbacks beta if it is rejected", async () => {
    const bad = Anthropic.APIError.generate(400, { error: { message: "fallbacks not supported" } }, "bad", new Headers());
    const { client, calls } = fakeClient([bad, { content: [{ type: "text", text: "ok" }], stop_reason: "end_turn" }]);
    const out = await streamMessage(client, payload, { ...OPTS, model: "claude-opus-5-5" });
    expect(out.text).toBe("ok");
    expect(calls).toHaveLength(2);
    expect((calls[0] as { fallbacks?: string }).fallbacks).toBe("default");
    expect((calls[1] as { fallbacks?: string }).fallbacks).toBeUndefined();
  });

  it("a 400 that the retry without fallbacks doesn't fix keeps the fallbacks for later requests", async () => {
    // e.g. «credit balance is too low» or «prompt is too long»: nothing to do with the beta.
    const bad = () => Anthropic.APIError.generate(400, { error: { message: "Your credit balance is too low" } }, "bad", new Headers());
    const { client, calls } = fakeClient([bad(), bad(), { content: [{ type: "text", text: "ok" }], stop_reason: "end_turn" }]);
    const opts = { ...OPTS, model: "claude-fable-5-1" };
    await expect(streamMessage(client, payload, opts)).rejects.toBeInstanceOf(Anthropic.BadRequestError);
    expect(calls).toHaveLength(2);
    await streamMessage(client, payload, opts);
    expect((calls[2] as { fallbacks?: string }).fallbacks).toBe("default");
  });
});

describe("describeError", () => {
  const h = new Headers();
  it("maps typed SDK errors to Russian messages", () => {
    expect(describeError(Anthropic.APIError.generate(401, {}, "x", h))).toContain("Ключ API не принят");
    expect(describeError(Anthropic.APIError.generate(429, {}, "x", h))).toContain("лимит");
    expect(describeError(Anthropic.APIError.generate(529, {}, "x", h))).toContain("перегружен");
    expect(describeError(Anthropic.APIError.generate(404, {}, "x", h), "claude-x")).toContain("«claude-x»");
    expect(describeError(new Anthropic.APIConnectionError({ message: "down" }))).toContain("api.anthropic.com");
    expect(describeError(new MissingKeyError())).toContain("API-ключ");
    expect(describeError(new RefusalError("cyber", null))).toContain("cyber");
  });
});

describe("Copilot live turns", () => {
  function setup(replies: Array<FakeMessage | Error>) {
    const conv = new Conversation(SYSTEM_PROMPT, "<meeting_context/>", SPEAKERS);
    const { client, calls } = fakeClient(replies);
    const copilot = new Copilot(conv, {
      getClient: () => client,
      getConfig: () => ({ model: "claude-opus-5", hintEffort: "low", summaryEffort: "high" }),
    });
    return { conv, copilot, calls };
  }

  it("commits successful replies and rolls back failed ones", async () => {
    const bad = Anthropic.APIError.generate(429, {}, "slow down", new Headers());
    const { conv, copilot } = setup([
      { content: [{ type: "text", text: "первый" }], stop_reason: "end_turn" },
      bad,
      { content: [{ type: "text", text: "третий" }], stop_reason: "end_turn" },
    ]);
    const spec = hintTask({ lang: "auto", speakers: SPEAKERS });
    await copilot.runLive(spec);
    await expect(copilot.runLive(spec)).rejects.toBeInstanceOf(Anthropic.RateLimitError);
    expect(conv.exchanges).toBe(1);
    expect(conv.awaitingReply).toBe(false);
    await copilot.runLive(spec);
    expect(conv.exchanges).toBe(2);
  });

  it("does not commit a refusal", async () => {
    const { conv, copilot } = setup([{ content: [], stop_reason: "refusal", stop_details: { category: null, explanation: null } }]);
    await expect(copilot.runLive(hintTask({ lang: "auto", speakers: SPEAKERS }))).rejects.toBeInstanceOf(RefusalError);
    expect(conv.exchanges).toBe(0);
    expect(conv.awaitingReply).toBe(false);
  });

  it("serializes live turns", async () => {
    const { conv, copilot, calls } = setup([
      { content: [{ type: "text", text: "a" }], stop_reason: "end_turn" },
      { content: [{ type: "text", text: "b" }], stop_reason: "end_turn" },
    ]);
    const spec = hintTask({ lang: "auto", speakers: SPEAKERS });
    await Promise.all([copilot.runLive(spec), copilot.runLive(spec)]);
    expect(conv.exchanges).toBe(2);
    const second = calls[1] as { messages: unknown[] };
    expect(second.messages).toHaveLength(3);
  });

  it("fails fast without a key and leaves the log untouched", async () => {
    const conv = new Conversation(SYSTEM_PROMPT, "<meeting_context/>", SPEAKERS);
    const copilot = new Copilot(conv, {
      getClient: () => null,
      getConfig: () => ({ model: "claude-opus-5", hintEffort: "low", summaryEffort: "high" }),
    });
    await expect(copilot.runLive(hintTask({ lang: "auto", speakers: SPEAKERS }))).rejects.toBeInstanceOf(MissingKeyError);
    expect(conv.awaitingReply).toBe(false);
  });
});

describe("structured outputs", () => {
  const format = { type: "json_schema" as const, schema: { type: "object", properties: {}, additionalProperties: false } };

  it("adds output_config.format next to effort for models that support it", () => {
    const p = buildParams(payload, { model: "claude-opus-5", effort: "low", maxTokens: 4000, fallbacks: false, format });
    expect(p.output_config).toEqual({ effort: "low", format });
    expect(p.fallbacks).toBeUndefined();
    const haiku = buildParams(payload, { model: "claude-haiku-4-5", effort: "low", maxTokens: 4000, format });
    expect(haiku.output_config).toEqual({ format });
    const old = buildParams(payload, { model: "claude-3-5-sonnet-20241022", effort: "low", maxTokens: 4000, format });
    expect(old.output_config).toBeUndefined();
  });

  it("retries once without the format if the model rejects it", async () => {
    const bad = Anthropic.APIError.generate(400, { error: { message: "output_config.format not supported" } }, "bad", new Headers());
    const { client, calls } = fakeClient([bad, { content: [{ type: "text", text: '{"suggestions":[]}' }], stop_reason: "end_turn" }]);
    const out = await streamMessage(client, payload, { model: "claude-sonnet-4-5", effort: "low", maxTokens: 4000, fallbacks: false, format });
    expect(out.text).toBe('{"suggestions":[]}');
    expect((calls[0] as { output_config?: { format?: unknown } }).output_config?.format).toEqual(format);
    expect((calls[1] as { output_config?: { format?: unknown } }).output_config?.format).toBeUndefined();
  });

  it("an unrelated 400 doesn't switch structured outputs off for the model", async () => {
    const bad = () => Anthropic.APIError.generate(400, { error: { message: "Your credit balance is too low" } }, "bad", new Headers());
    const { client, calls } = fakeClient([bad(), bad(), { content: [{ type: "text", text: "{}" }], stop_reason: "end_turn" }]);
    const opts = { model: "claude-sonnet-5", effort: "low" as const, maxTokens: 4000, fallbacks: false, format };
    await expect(streamMessage(client, payload, opts)).rejects.toBeInstanceOf(Anthropic.BadRequestError);
    await streamMessage(client, payload, opts);
    expect((calls[2] as { output_config?: { format?: unknown } }).output_config?.format).toEqual(format);
  });
});

describe("Copilot side requests", () => {
  it("run off the conversation: the log and its pending lines are untouched", async () => {
    const conv = new Conversation(SYSTEM_PROMPT, "<meeting_context/>", SPEAKERS);
    conv.addFinal({ id: "system-1", source: "system", speaker: "sys:1", startMs: 0, endMs: 1000, text: "айдос что скажешь", isFinal: true });
    const { client, calls } = fakeClient([{ content: [{ type: "text", text: "{}" }], stop_reason: "end_turn" }]);
    const copilot = new Copilot(conv, {
      getClient: () => client,
      getConfig: () => ({ model: "claude-opus-5", hintEffort: "medium", summaryEffort: "high" }),
    });
    const side = { system: [{ type: "text" as const, text: "side" }], messages: [{ role: "user" as const, content: "x" }] };
    await copilot.runSide(side, { effort: "low", maxTokens: 4000, fallbacks: false });
    expect(conv.exchanges).toBe(0);
    expect(conv.pendingCount()).toBe(1);
    expect(calls[0]).toMatchObject({ model: "claude-opus-5", system: side.system, output_config: { effort: "low" } });
  });
});
