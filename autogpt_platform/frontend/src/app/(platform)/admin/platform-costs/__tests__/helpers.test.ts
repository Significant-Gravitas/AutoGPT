import { describe, expect, it } from "vitest";
import type { ProviderCostSummary } from "@/app/api/__generated__/models/providerCostSummary";
import {
  toDateOrUndefined,
  formatMicrodollars,
  formatTokens,
  formatDuration,
  estimateCostForRow,
  trackingValue,
} from "../helpers";

function makeRow(overrides: Partial<ProviderCostSummary>): ProviderCostSummary {
  return {
    provider: "openai",
    tracking_type: null,
    total_cost_microdollars: 0,
    total_input_tokens: 0,
    total_output_tokens: 0,
    total_duration_seconds: 0,
    request_count: 0,
    ...overrides,
  };
}

describe("toDateOrUndefined", () => {
  it("returns undefined for empty string", () => {
    expect(toDateOrUndefined("")).toBeUndefined();
  });

  it("returns undefined for undefined", () => {
    expect(toDateOrUndefined(undefined)).toBeUndefined();
  });

  it("returns undefined for invalid date string", () => {
    expect(toDateOrUndefined("not-a-date")).toBeUndefined();
  });

  it("returns a Date for a valid ISO string", () => {
    const result = toDateOrUndefined("2026-01-15T00:00:00Z");
    expect(result).toBeInstanceOf(Date);
    expect(result!.toISOString()).toBe("2026-01-15T00:00:00.000Z");
  });
});

describe("formatMicrodollars", () => {
  it("formats zero", () => {
    expect(formatMicrodollars(0)).toBe("$0.0000");
  });

  it("formats a small amount", () => {
    expect(formatMicrodollars(50_000)).toBe("$0.0500");
  });

  it("formats one dollar", () => {
    expect(formatMicrodollars(1_000_000)).toBe("$1.0000");
  });
});

describe("formatTokens", () => {
  it("formats small numbers as-is", () => {
    expect(formatTokens(500)).toBe("500");
  });

  it("formats thousands with K suffix", () => {
    expect(formatTokens(1_500)).toBe("1.5K");
  });

  it("formats millions with M suffix", () => {
    expect(formatTokens(2_500_000)).toBe("2.5M");
  });
});

describe("formatDuration", () => {
  it("formats seconds", () => {
    expect(formatDuration(30)).toBe("30.0s");
  });

  it("formats minutes", () => {
    expect(formatDuration(90)).toBe("1.5m");
  });

  it("formats hours", () => {
    expect(formatDuration(5400)).toBe("1.5h");
  });
});

describe("estimateCostForRow", () => {
  it("returns microdollars directly for cost_usd tracking", () => {
    const row = makeRow({
      tracking_type: "cost_usd",
      total_cost_microdollars: 500_000,
    });
    expect(estimateCostForRow(row, {})).toBe(500_000);
  });

  it("returns reported cost for token tracking when cost > 0", () => {
    const row = makeRow({
      tracking_type: "tokens",
      total_cost_microdollars: 100_000,
      total_input_tokens: 1000,
      total_output_tokens: 500,
    });
    expect(estimateCostForRow(row, {})).toBe(100_000);
  });

  it("estimates cost from default rate for token tracking with zero cost", () => {
    const row = makeRow({
      provider: "openai",
      tracking_type: "tokens",
      total_cost_microdollars: 0,
      total_input_tokens: 500,
      total_output_tokens: 500,
    });
    // 1000 tokens / 1000 * 0.005 USD * 1_000_000 = 5000
    expect(estimateCostForRow(row, {})).toBe(5000);
  });

  it("returns null for unknown token provider with zero cost", () => {
    const row = makeRow({
      provider: "unknown_provider",
      tracking_type: "tokens",
      total_cost_microdollars: 0,
    });
    expect(estimateCostForRow(row, {})).toBeNull();
  });

  it("uses per-run override when provided", () => {
    const row = makeRow({
      provider: "google_maps",
      tracking_type: "per_run",
      request_count: 10,
    });
    // override = 0.05 * 10 * 1_000_000 = 500_000
    expect(estimateCostForRow(row, { google_maps: 0.05 })).toBe(500_000);
  });

  it("uses default per-run cost when no override", () => {
    const row = makeRow({
      provider: "google_maps",
      tracking_type: null,
      request_count: 5,
    });
    // 0.032 * 5 * 1_000_000 = 160_000
    expect(estimateCostForRow(row, {})).toBe(160_000);
  });

  it("returns null for unknown per_run provider", () => {
    const row = makeRow({
      provider: "totally_unknown",
      tracking_type: "per_run",
      request_count: 3,
    });
    expect(estimateCostForRow(row, {})).toBeNull();
  });

  it("returns cost for other tracking types when cost > 0", () => {
    const row = makeRow({
      tracking_type: "duration_seconds",
      total_cost_microdollars: 42_000,
    });
    expect(estimateCostForRow(row, {})).toBe(42_000);
  });

  it("returns null for other tracking types when cost is 0", () => {
    const row = makeRow({
      tracking_type: "duration_seconds",
      total_cost_microdollars: 0,
    });
    expect(estimateCostForRow(row, {})).toBeNull();
  });
});

describe("trackingValue", () => {
  it("returns formatted microdollars for cost_usd", () => {
    const row = makeRow({
      tracking_type: "cost_usd",
      total_cost_microdollars: 1_000_000,
    });
    expect(trackingValue(row)).toBe("$1.0000");
  });

  it("returns formatted token count for tokens", () => {
    const row = makeRow({
      tracking_type: "tokens",
      total_input_tokens: 500,
      total_output_tokens: 500,
    });
    expect(trackingValue(row)).toBe("1.0K");
  });

  it("returns formatted duration for duration_seconds", () => {
    const row = makeRow({
      tracking_type: "duration_seconds",
      total_duration_seconds: 120,
    });
    expect(trackingValue(row)).toBe("2.0m");
  });

  it("returns run count for per_run (default tracking)", () => {
    const row = makeRow({
      tracking_type: null,
      request_count: 42,
    });
    expect(trackingValue(row)).toBe("42 runs");
  });

  it("returns formatted token count for characters tracking", () => {
    const row = makeRow({
      tracking_type: "characters",
      total_input_tokens: 2000,
      total_output_tokens: 500,
    });
    expect(trackingValue(row)).toBe("2.5K");
  });

  it("returns formatted duration for sandbox_seconds", () => {
    const row = makeRow({
      tracking_type: "sandbox_seconds",
      total_duration_seconds: 7200,
    });
    expect(trackingValue(row)).toBe("2.0h");
  });

  it("returns formatted duration for walltime_seconds", () => {
    const row = makeRow({
      tracking_type: "walltime_seconds",
      total_duration_seconds: 45,
    });
    expect(trackingValue(row)).toBe("45.0s");
  });
});
