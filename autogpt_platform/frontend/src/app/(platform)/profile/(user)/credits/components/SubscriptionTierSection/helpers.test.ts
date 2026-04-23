import { describe, expect, it } from "vitest";

import {
  formatCost,
  formatPendingDate,
  getTierLabel,
  TIERS,
  TIER_ORDER,
} from "./helpers";

describe("formatCost", () => {
  it("returns 'Free' for the FREE tier regardless of cents", () => {
    expect(formatCost(0, "FREE")).toBe("Free");
    expect(formatCost(999, "FREE")).toBe("Free");
  });

  it("returns a placeholder when paid tier has no price yet", () => {
    expect(formatCost(0, "PRO")).toBe("Pricing available soon");
    expect(formatCost(0, "BUSINESS")).toBe("Pricing available soon");
  });

  it("formats cents to a dollars-per-month string for paid tiers", () => {
    expect(formatCost(999, "PRO")).toBe("$9.99/mo");
    expect(formatCost(4900, "BUSINESS")).toBe("$49.00/mo");
  });
});

describe("getTierLabel", () => {
  it("returns the canonical label for known tiers", () => {
    expect(getTierLabel("FREE")).toBe("Free");
    expect(getTierLabel("PRO")).toBe("Pro");
    expect(getTierLabel("BUSINESS")).toBe("Business");
  });

  it("title-cases unknown tier keys as a fallback", () => {
    // ENTERPRISE is in TIER_ORDER but intentionally not in TIERS — it still
    // needs a readable label so the pending-change sentence reads cleanly.
    expect(getTierLabel("ENTERPRISE")).toBe("Enterprise");
    expect(getTierLabel("CUSTOM_TIER")).toBe("Custom_tier");
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
