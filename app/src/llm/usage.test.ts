import { describe, expect, it } from "vitest";
import { addUsage, cacheHitRate, costOf, EMPTY_USAGE, formatCost, formatTokens, parseUsage, priceFor } from "./usage";

const u = (input: number, cacheRead: number, cacheWrite: number, output: number) => ({ input, cacheRead, cacheWrite, output });

describe("usage and cost", () => {
  it("prices the current models from the docs' table; unknown ones have no price", () => {
    expect(priceFor("claude-opus-5")).toEqual({ input: 5, output: 25, cacheRead: 0.5 });
    expect(priceFor("claude-opus-5-5")).toEqual({ input: 4, output: 20, cacheRead: 0.2 });
    expect(priceFor("claude-opus-4-8")?.input).toBe(5);
    expect(priceFor("claude-fable-5-1")).toEqual({ input: 10, output: 50, cacheRead: 0.25 });
    expect(priceFor("claude-fable-5")?.cacheRead).toBe(1);
    expect(priceFor("claude-sonnet-5")).toEqual({ input: 2, output: 10, cacheRead: 0.2 });
    expect(priceFor("claude-sonnet-4-6")?.input).toBe(3);
    expect(priceFor("claude-haiku-4-5-20251001")).toEqual({ input: 1, output: 5, cacheRead: 0.1 });
    expect(priceFor("claude-opus-4-5")).toBeNull();
    expect(priceFor("my-proxy")).toBeNull();
  });

  it("costs uncached input, 1.25× cache writes, cheap cache reads and output", () => {
    // Opus 5: 1M uncached = $5, 1M written = $6.25, 1M read = $0.50, 1M out = $25.
    expect(costOf("claude-opus-5", u(1e6, 0, 0, 0))).toBeCloseTo(5);
    expect(costOf("claude-opus-5", u(0, 0, 1e6, 0))).toBeCloseTo(6.25);
    expect(costOf("claude-opus-5", u(0, 1e6, 0, 0))).toBeCloseTo(0.5);
    expect(costOf("claude-opus-5", u(0, 0, 0, 1e6))).toBeCloseTo(25);
    expect(costOf("unknown", u(1, 1, 1, 1))).toBeNull();
  });

  it("aggregates per meeting: hit rate, peak prompt of the live log, unpriced requests", () => {
    let t = addUsage(EMPTY_USAGE, "claude-opus-5", u(3_000, 0, 5_000, 200), true);
    t = addUsage(t, "claude-opus-5", u(300, 8_000, 600, 150), true);
    t = addUsage(t, "claude-opus-5", u(20_000, 0, 20_000, 3_000)); // final summary: not the live log
    t = addUsage(t, "mystery-model", u(10, 0, 0, 10));
    expect(t).toMatchObject({ requests: 4, input: 23_310, cacheRead: 8_000, cacheWrite: 25_600, output: 3_360, unpriced: 1, peakPrompt: 8_900 });
    expect(cacheHitRate(t)).toBeCloseTo(8_000 / (23_310 + 8_000 + 25_600));
    expect(cacheHitRate(EMPTY_USAGE)).toBeNull();
    expect(t.cost).toBeCloseTo(
      (3_000 * 5 + 5_000 * 6.25 + 200 * 25 + 300 * 5 + 8_000 * 0.5 + 600 * 6.25 + 150 * 25 + 20_000 * 5 + 20_000 * 6.25 + 3_000 * 25) / 1e6,
    );
  });

  it("formats compactly and validates stored totals", () => {
    expect(formatTokens(950)).toBe("950");
    expect(formatTokens(1_234)).toBe("1,2K");
    expect(formatTokens(184_000)).toBe("184K");
    expect(formatTokens(2_500_000)).toBe("2,5M");
    expect(formatCost(0)).toBe("$0");
    expect(formatCost(0.004)).toBe("<$0.01");
    expect(formatCost(0.4123)).toBe("$0.41");
    expect(formatCost(12.34)).toBe("$12.3");
    expect(parseUsage({ ...EMPTY_USAGE, requests: 3 })?.requests).toBe(3);
    expect(parseUsage({ requests: "3" })).toBeNull();
    expect(parseUsage(null)).toBeNull();
  });
});
