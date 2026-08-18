import { describe, expect, it } from "vitest";
import {
  formatBytes,
  formatCents,
  formatMicrodollarsAsUsd,
  formatResetTime,
  formatTierLabel,
  isUsageExhausted,
} from "../usageHelpers";

describe("formatCents", () => {
  it("formats whole dollars", () => {
    expect(formatCents(500)).toBe("$5.00");
  });

  it("formats zero", () => {
    expect(formatCents(0)).toBe("$0.00");
  });

  it("formats fractional cents", () => {
    expect(formatCents(1999)).toBe("$19.99");
  });
});

describe("formatMicrodollarsAsUsd", () => {
  it("formats zero as $0.00", () => {
    expect(formatMicrodollarsAsUsd(0)).toBe("$0.00");
  });

  it("formats whole dollar amounts", () => {
    expect(formatMicrodollarsAsUsd(1_500_000)).toBe("$1.50");
  });

  it("formats amounts that round to $0.00 but are > 0 as <$0.01", () => {
    expect(formatMicrodollarsAsUsd(999)).toBe("<$0.01");
  });

  it("formats exactly one cent as $0.01", () => {
    expect(formatMicrodollarsAsUsd(10_000)).toBe("$0.01");
  });

  it("formats negative input with toFixed semantics (no special case)", () => {
    // Negative should never come from the backend, but the helper is
    // safe — it simply passes through `toFixed`.
    expect(formatMicrodollarsAsUsd(-1_500_000)).toBe("$-1.50");
  });

  it("formats very large values without truncating", () => {
    expect(formatMicrodollarsAsUsd(1_234_567_890)).toBe("$1234.57");
  });
});

describe("formatResetTime", () => {
  it("returns 'now' when reset time is in the past", () => {
    const now = new Date("2026-04-21T12:00:00Z");
    const past = new Date("2026-04-21T11:59:00Z");
    expect(formatResetTime(past, now)).toBe("now");
  });

  it("renders sub-hour resets as minutes", () => {
    const now = new Date("2026-04-21T12:00:00Z");
    const future = new Date("2026-04-21T12:15:00Z");
    expect(formatResetTime(future, now)).toBe("in 15m");
  });

  it("renders same-day resets as 'Xh Ym'", () => {
    const now = new Date("2026-04-21T12:00:00Z");
    const future = new Date("2026-04-21T14:30:00Z");
    expect(formatResetTime(future, now)).toBe("in 2h 30m");
  });

  it("renders future-day resets as a localized date string", () => {
    const now = new Date("2026-04-21T12:00:00Z");
    const future = new Date("2026-04-24T12:00:00Z");
    // Not asserting exact format (localized), just that it's not the
    // minute/hour form.
    expect(formatResetTime(future, now)).not.toMatch(/^in \d/);
  });
});

describe("formatTierLabel", () => {
  it("returns null for null", () => {
    expect(formatTierLabel(null)).toBeNull();
  });

  it("returns null for undefined", () => {
    expect(formatTierLabel(undefined)).toBeNull();
  });

  it("returns null for empty string", () => {
    expect(formatTierLabel("")).toBeNull();
  });

  it("returns null for the NO_TIER sentinel", () => {
    expect(formatTierLabel("NO_TIER")).toBeNull();
  });

  it("capitalizes a known tier", () => {
    expect(formatTierLabel("PRO")).toBe("Pro");
    expect(formatTierLabel("BASIC")).toBe("Basic");
  });

  it("normalizes first-letter casing for lowercase or mixed-case input", () => {
    expect(formatTierLabel("pro")).toBe("Pro");
    expect(formatTierLabel("pRo")).toBe("Pro");
  });
});

describe("isUsageExhausted", () => {
  it("returns false for null/undefined usage", () => {
    expect(isUsageExhausted(null)).toBe(false);
    expect(isUsageExhausted(undefined)).toBe(false);
  });

  it("returns false when neither window is over 100%", () => {
    expect(
      isUsageExhausted({
        daily: { percent_used: 50 },
        weekly: { percent_used: 60 },
      }),
    ).toBe(false);
  });

  it("returns true when daily is exhausted", () => {
    expect(
      isUsageExhausted({
        daily: { percent_used: 100 },
        weekly: { percent_used: 10 },
      }),
    ).toBe(true);
  });

  it("returns true when weekly is exhausted", () => {
    expect(
      isUsageExhausted({
        daily: { percent_used: 0 },
        weekly: { percent_used: 100 },
      }),
    ).toBe(true);
  });

  it("treats missing percent_used as 0", () => {
    expect(isUsageExhausted({ daily: null, weekly: null })).toBe(false);
    expect(isUsageExhausted({})).toBe(false);
  });
});

describe("formatBytes", () => {
  it.each([
    [0, "0 B"],
    [512, "512 B"],
    [1024, "1 KB"],
    [250 * 1024, "250 KB"],
    [1023 * 1024, "1023 KB"],
    [1024 * 1024, "1 MB"],
    [250 * 1024 * 1024, "250 MB"],
    [1024 * 1024 * 1024, "1.0 GB"],
    [5 * 1024 * 1024 * 1024, "5.0 GB"],
  ])("formats %d bytes as %s", (input, expected) => {
    expect(formatBytes(input)).toBe(expected);
  });
});
