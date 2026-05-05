import { describe, expect, it } from "vitest";

import {
  formatCost,
  formatPendingDate,
  formatRelativeMultiplier,
  getTierLabel,
  TIERS,
  TIER_ORDER,
} from "./helpers";

describe("formatCost", () => {
  it("returns 'Free' for any tier with cost = 0", () => {
    expect(formatCost(0, "BASIC")).toBe("Free");
    expect(formatCost(0, "PRO")).toBe("Free");
    expect(formatCost(0, "MAX")).toBe("Free");
    expect(formatCost(0, "BUSINESS")).toBe("Free");
  });

  it("formats cents to a dollars-per-month string for any non-zero cost", () => {
    expect(formatCost(999, "BASIC")).toBe("$9.99/mo");
    expect(formatCost(999, "PRO")).toBe("$9.99/mo");
    expect(formatCost(32000, "MAX")).toBe("$320.00/mo");
    expect(formatCost(4900, "BUSINESS")).toBe("$49.00/mo");
  });
});

describe("getTierLabel", () => {
  it("returns the canonical label for known tiers", () => {
    expect(getTierLabel("BASIC")).toBe("Basic");
    expect(getTierLabel("PRO")).toBe("Pro");
    expect(getTierLabel("MAX")).toBe("Max");
    expect(getTierLabel("BUSINESS")).toBe("Business");
  });

  it("title-cases unknown tier keys as a fallback", () => {
    // ENTERPRISE is in TIER_ORDER but intentionally not in TIERS — it still
    // needs a readable label so the pending-change sentence reads cleanly.
    expect(getTierLabel("ENTERPRISE")).toBe("Enterprise");
    expect(getTierLabel("CUSTOM_TIER")).toBe("Custom_tier");
  });

  it("renders NO_TIER as 'No subscription' (not the title-case fallback)", () => {
    // The fallback would produce "No_tier" — surfaced in the pending-change
    // banner during cancel-at-period-end as "Scheduled to downgrade to
    // No_tier on …". Special-cased here so the user-facing copy matches the
    // semantic of the value.
    expect(getTierLabel("NO_TIER")).toBe("No subscription");
  });
});

describe("formatPendingDate", () => {
  it("formats a Date into a stable en-US string", () => {
    const d = new Date("2026-03-15T12:00:00Z");
    // en-US pinned to avoid SSR/CSR hydration mismatch, so the output must
    // match regardless of the host locale.
    expect(formatPendingDate(d)).toBe("Mar 15, 2026");
  });

  it("accepts an ISO string and produces the same output as a Date", () => {
    expect(formatPendingDate("2026-03-15T12:00:00Z")).toBe("Mar 15, 2026");
  });
});

describe("TIERS / TIER_ORDER", () => {
  it("every entry in TIERS has a key that appears in TIER_ORDER", () => {
    for (const tier of TIERS) {
      expect(TIER_ORDER).toContain(tier.key);
    }
  });
});

describe("formatRelativeMultiplier", () => {
  it("returns null for the lowest visible tier — it's the baseline", () => {
    expect(
      formatRelativeMultiplier("BASIC", { BASIC: 1, PRO: 5, MAX: 20 }),
    ).toBeNull();
  });

  it("formats a clean integer multiplier as 'N.0x rate limits'", () => {
    expect(formatRelativeMultiplier("PRO", { BASIC: 1, PRO: 4, MAX: 20 })).toBe(
      "4.0x rate limits",
    );
  });

  it("rounds to one decimal when the ratio isn't a whole number", () => {
    expect(formatRelativeMultiplier("MAX", { BASIC: 2, PRO: 5, MAX: 17 })).toBe(
      "8.5x rate limits",
    );
  });

  it("returns null when the tier isn't in the multipliers map (hidden)", () => {
    expect(
      formatRelativeMultiplier("BUSINESS", { BASIC: 1, PRO: 5 }),
    ).toBeNull();
  });

  it("ignores the baseline from hidden tiers so visible-tier deltas stay honest", () => {
    // BASIC hidden but PRO/MAX visible — PRO becomes the baseline, MAX is 4×.
    expect(formatRelativeMultiplier("MAX", { PRO: 5, MAX: 20 })).toBe(
      "4.0x rate limits",
    );
    expect(formatRelativeMultiplier("PRO", { PRO: 5, MAX: 20 })).toBeNull();
  });

  it("handles fractional LD-provided multipliers cleanly", () => {
    // LD override can set e.g. PRO=7.5×; the relative display still computes
    // correctly against a non-integer minimum.
    expect(formatRelativeMultiplier("PRO", { BASIC: 1.5, PRO: 7.5 })).toBe(
      "5.0x rate limits",
    );
  });

  it("returns null for every tier when all visible multipliers are equal", () => {
    // Edge case: if LD sets every tier to the same value, none are "above"
    // the baseline — the UI shouldn't label any of them.
    expect(formatRelativeMultiplier("PRO", { PRO: 5, MAX: 5 })).toBeNull();
    expect(formatRelativeMultiplier("MAX", { PRO: 5, MAX: 5 })).toBeNull();
  });

  it("returns null when the tier's own multiplier is zero or negative", () => {
    // Defensive: a misconfigured LD value leaking through shouldn't render as
    // "0.0x rate limits" — hide the badge entirely.
    expect(formatRelativeMultiplier("BASIC", { BASIC: 0, PRO: 5 })).toBeNull();
    expect(formatRelativeMultiplier("BASIC", { BASIC: -1, PRO: 5 })).toBeNull();
  });

  it("rounds 8.533... to '8.5x rate limits' (not 8.53 or 9)", () => {
    // Price-ratio-derived multiplier from the real $320/$50 → 6.4, or from
    // limit-ratio 26.67/3.13 → ~8.53. The display rule is one decimal place.
    expect(formatRelativeMultiplier("MAX", { PRO: 3, MAX: 25.6 })).toBe(
      "8.5x rate limits",
    );
  });
});
