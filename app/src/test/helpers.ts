import type Anthropic from "@anthropic-ai/sdk";

/** In-memory `localStorage` for tests running in node. */
export class MemoryStorage implements Storage {
  private map = new Map<string, string>();
  get length() {
    return this.map.size;
  }
  clear() {
    this.map.clear();
  }
  getItem(key: string) {
    return this.map.has(key) ? this.map.get(key)! : null;
  }
  key(i: number) {
    return [...this.map.keys()][i] ?? null;
  }
  removeItem(key: string) {
    this.map.delete(key);
  }
  setItem(key: string, value: string) {
    this.map.set(key, String(value));
  }
}

export type CapturedRequest = {
  model: string;
  system: Array<{ text: string }>;
  messages: Array<{ role: string; content: Array<{ type: string; text: string }> | string }>;
  output_config?: { effort?: string; format?: unknown };
  [k: string]: unknown;
};

/** All text of a captured request, for `toContain` checks. */
export function requestText(r: CapturedRequest): string {
  const parts = r.messages.flatMap((m) => (typeof m.content === "string" ? [m.content] : m.content.map((b) => b.text)));
  return [...r.system.map((s) => s.text), ...parts].join("\n");
}

/** The last user block of a request: the task. */
export function taskOf(r: CapturedRequest): string {
  const last = r.messages.at(-1)!;
  return typeof last.content === "string" ? last.content : last.content.at(-1)!.text;
}

/**
 * Stand-in for `client.beta.messages.stream()`: `reply(params)` decides the text, which is
 * streamed as a couple of deltas before `finalMessage()` resolves.
 */
export type FakeUsage = { input_tokens: number; output_tokens: number; cache_read_input_tokens: number; cache_creation_input_tokens: number };

export function fakeClaude(reply: (req: CapturedRequest) => string, usage?: (req: CapturedRequest, index: number) => Partial<FakeUsage>) {
  const calls: CapturedRequest[] = [];
  const client = {
    beta: {
      messages: {
        stream(params: CapturedRequest) {
          calls.push(params);
          const index = calls.length - 1;
          const text = reply(params);
          const handlers: Array<(d: string) => void> = [];
          return {
            on(event: string, fn: (d: string) => void) {
              if (event === "text") handlers.push(fn);
              return this;
            },
            async finalMessage() {
              await new Promise((r) => setTimeout(r, 1));
              const mid = Math.ceil(text.length / 2);
              for (const d of [text.slice(0, mid), text.slice(mid)]) if (d) handlers.forEach((h) => h(d));
              return {
                model: params.model,
                stop_reason: "end_turn",
                stop_details: null,
                content: [{ type: "text", text }],
                usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 0, cache_creation_input_tokens: 0, ...usage?.(params, index) },
              };
            },
          };
        },
      },
    },
  };
  return { client: client as unknown as Anthropic, calls };
}

export async function waitFor(cond: () => boolean, timeoutMs = 10_000, what = "condition"): Promise<void> {
  const start = Date.now();
  while (!cond()) {
    if (Date.now() - start > timeoutMs) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 5));
  }
}
