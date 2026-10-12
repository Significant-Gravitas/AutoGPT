import { describe, expect, it } from "vitest";
import {
  barPercent,
  compactionLabel,
  formatTokens,
  parseCompactionOutput,
  phaseProgress,
  REDUCED_MOTION_STEP_PERCENT,
  tauForTokens,
} from "../helpers";

describe("phaseProgress", () => {
  it("starts at the base value", () => {
    expect(phaseProgress(0.02, 0.55, 0, 15_000)).toBeCloseTo(0.02, 5);
  });

  it("never reaches the cap", () => {
    expect(phaseProgress(0.02, 0.55, 10 * 60 * 1000, 15_000)).toBeLessThan(
      0.55,
    );
  });

  it("is monotonically increasing", () => {
    let prev = -1;
    for (let t = 0; t <= 120_000; t += 1_000) {
      const p = phaseProgress(0.02, 0.55, t, 15_000);
      expect(p).toBeGreaterThan(prev);
      prev = p;
    }
  });

  it("reaches ~63% of the span at one time constant", () => {
    expect(phaseProgress(0, 1, 15_000, 15_000)).toBeCloseTo(0.632, 2);
  });
});

describe("tauForTokens", () => {
  it("floors at 12s for unknown or small contexts", () => {
    expect(tauForTokens(undefined)).toBe(12_000);
    expect(tauForTokens(1_000)).toBe(12_000);
  });

  it("grows with context size", () => {
    expect(tauForTokens(200_000)).toBeGreaterThan(tauForTokens(50_000));
  });

  it("caps at 45s", () => {
    expect(tauForTokens(5_000_000)).toBe(45_000);
  });
});

describe("parseCompactionOutput", () => {
  it("reads the JSON payload", () => {
    const stats = parseCompactionOutput(
      JSON.stringify({
        summary:
          "Earlier messages were summarized to fit within context limits.",
        tokensBefore: 128_000,
        tokensAfter: 31_000,
        messagesBefore: 412,
        messagesAfter: 38,
      }),
    );
    expect(stats.tokensBefore).toBe(128_000);
    expect(stats.messagesAfter).toBe(38);
  });

  it("accepts an already-parsed object", () => {
    const stats = parseCompactionOutput({ summary: "x", tokensBefore: 5 });
    expect(stats.tokensBefore).toBe(5);
  });

  it("reads the dropped flag", () => {
    const stats = parseCompactionOutput(
      JSON.stringify({ summary: "dropped", dropped: true, messagesBefore: 9 }),
    );
    expect(stats).toEqual({ dropped: true, messagesBefore: 9 });
  });

  it("yields no stats for the legacy plain sentence", () => {
    const legacy =
      "Earlier messages were summarized to fit within context limits.";
    expect(parseCompactionOutput(legacy)).toEqual({});
  });

  it("survives junk", () => {
    expect(parseCompactionOutput(undefined)).toEqual({});
    expect(parseCompactionOutput(42)).toEqual({});
  });

  it("drops implausible counts — zeros and fractions never reach the copy", () => {
    const stats = parseCompactionOutput({
      summary: "x",
      tokensBefore: 0,
      tokensAfter: 0,
      messagesBefore: 412.5,
      messagesAfter: 38,
    });
    expect(stats).toEqual({ messagesAfter: 38 });
  });
});

describe("formatTokens", () => {
  it("abbreviates thousands", () => {
    expect(formatTokens(31_000)).toBe("31K");
    expect(formatTokens(128_000)).toBe("128K");
  });

  it("keeps small numbers exact", () => {
    expect(formatTokens(840)).toBe("840");
  });

  it("rounds to one decimal below 10K", () => {
    expect(formatTokens(4_200)).toBe("4.2K");
    expect(formatTokens(4_000)).toBe("4K");
  });

  it("is continuous at the 10K boundary", () => {
    expect(formatTokens(9_999)).toBe("10K");
    expect(formatTokens(10_000)).toBe("10K");
  });

  it("switches to millions instead of rendering 1000K", () => {
    expect(formatTokens(999_499)).toBe("999K");
    expect(formatTokens(999_500)).toBe("1M");
    expect(formatTokens(1_000_000)).toBe("1M");
    expect(formatTokens(1_500_000)).toBe("1.5M");
    expect(formatTokens(128_000_000)).toBe("128M");
  });
});

describe("barPercent", () => {
  it("paints the exact percent when motion is allowed", () => {
    expect(barPercent(0.02, false)).toBe(2);
    expect(barPercent(0.473, false)).toBe(47);
  });

  it("steps the fill under reduced motion instead of creeping", () => {
    expect(barPercent(0.473, true)).toBe(40);
    expect(barPercent(0.5, true)).toBe(50);
    expect(barPercent(0.59, true)).toBe(50);
  });

  it("never claims progress the curve has not made", () => {
    for (let p = 0; p <= 1.0001; p += 0.01) {
      const stepped = barPercent(p, true);
      expect(stepped).toBeLessThanOrEqual(barPercent(p, false));
      expect(stepped % REDUCED_MOTION_STEP_PERCENT).toBe(0);
    }
  });

  it("still reaches a full bar", () => {
    expect(barPercent(1, true)).toBe(100);
  });
});

describe("compactionLabel", () => {
  it("narrates each live phase", () => {
    expect(compactionLabel("summarizing", {})).toBe(
      "Condensing our conversation…",
    );
    expect(compactionLabel("rebuilding", {})).toBe("Reloading context…");
  });

  it("celebrates settled rows with real numbers when it has them", () => {
    expect(
      compactionLabel(null, {
        tokensBefore: 128_000,
        tokensAfter: 31_000,
        messagesBefore: 412,
        messagesAfter: 38,
      }),
    ).toBe("Condensed 412 messages · 128K → 31K tokens");
  });

  it("settles with tokens only when no messages were removed", () => {
    expect(
      compactionLabel(null, { tokensBefore: 128_000, tokensAfter: 31_000 }),
    ).toBe("Condensed the conversation · 128K → 31K tokens");
  });

  it("skips the message count when content was summarized in place", () => {
    // Observed in production: 60 messages in, 60 messages out — the rows
    // survived, only their content shrank. "Condensed 60 messages" would
    // read as removal that never happened.
    expect(
      compactionLabel(null, {
        tokensBefore: 128_000,
        tokensAfter: 31_000,
        messagesBefore: 60,
        messagesAfter: 60,
      }),
    ).toBe("Condensed the conversation · 128K → 31K tokens");
  });

  it("refuses to advertise an inverted token measurement", () => {
    expect(
      compactionLabel(null, { tokensBefore: 31_000, tokensAfter: 128_000 }),
    ).toBe("Condensed the conversation to keep going");
  });

  it("degrades gracefully for legacy rows with no stats", () => {
    expect(compactionLabel(null, {})).toBe(
      "Condensed the conversation to keep going",
    );
  });
});
