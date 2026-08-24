import { describe, expect, it } from "vitest";
import {
  compactionLabel,
  finishProgress,
  formatTokens,
  parseCompactionOutput,
  phaseProgress,
  tauForTokens,
  FINISH_MS,
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

describe("finishProgress", () => {
  it("lands on exactly 1 at the finish duration", () => {
    expect(finishProgress(0.9, FINISH_MS)).toBeCloseTo(1, 1);
  });

  it("clamps past the finish duration", () => {
    expect(finishProgress(0.9, FINISH_MS * 10)).toBe(1);
  });

  it("starts from the base", () => {
    expect(finishProgress(0.42, 0)).toBeCloseTo(0.42, 5);
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
    const parsed = parseCompactionOutput(
      JSON.stringify({
        summary:
          "Earlier messages were summarized to fit within context limits.",
        tokensBefore: 128_000,
        tokensAfter: 31_000,
        messagesBefore: 412,
        messagesAfter: 38,
      }),
    );
    expect(parsed.stats.tokensBefore).toBe(128_000);
    expect(parsed.stats.messagesAfter).toBe(38);
  });

  it("accepts an already-parsed object", () => {
    const parsed = parseCompactionOutput({ summary: "x", tokensBefore: 5 });
    expect(parsed.summary).toBe("x");
    expect(parsed.stats.tokensBefore).toBe(5);
  });

  it("falls back to the legacy plain sentence", () => {
    const legacy =
      "Earlier messages were summarized to fit within context limits.";
    const parsed = parseCompactionOutput(legacy);
    expect(parsed.summary).toBe(legacy);
    expect(parsed.stats).toEqual({});
  });

  it("survives junk", () => {
    expect(parseCompactionOutput(undefined).stats).toEqual({});
    expect(parseCompactionOutput(42).stats).toEqual({});
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
  });
});

describe("compactionLabel", () => {
  it("narrates each live phase", () => {
    expect(compactionLabel("summarizing", {})).toBe(
      "Condensing our conversation…",
    );
    expect(compactionLabel("rebuilding", {})).toBe("Reloading context…");
  });

  it("celebrates with real numbers when it has them", () => {
    expect(
      compactionLabel("done", {
        tokensBefore: 128_000,
        tokensAfter: 31_000,
        messagesBefore: 412,
      }),
    ).toBe("Condensed 412 messages · 128K → 31K tokens");
  });

  it("settles without a phase", () => {
    expect(
      compactionLabel(null, { tokensBefore: 128_000, tokensAfter: 31_000 }),
    ).toBe("Condensed the conversation · 128K → 31K tokens");
  });

  it("degrades gracefully for legacy rows with no stats", () => {
    expect(compactionLabel(null, {})).toBe(
      "Condensed the conversation to keep going",
    );
  });
});
