import { describe, expect, it } from "vitest";
import {
  formatCents,
  formatMicrodollarsAsUsd,
  formatResetTime,
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
