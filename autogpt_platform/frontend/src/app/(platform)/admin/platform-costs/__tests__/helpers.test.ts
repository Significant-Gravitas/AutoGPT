import { describe, expect, it } from "vitest";
import type { ProviderCostSummary } from "@/app/api/__generated__/models/providerCostSummary";
import {
  toDateOrUndefined,
  formatMicrodollars,
  formatTokens,
  formatDuration,
  estimateCostForRow,
  trackingValue,
  toLocalInput,
  toUtcIso,
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
    expect(estimateCostForRow(row, { "google_maps:per_run": 0.05 })).toBe(
      500_000,
    );
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

  it("returns null for duration tracking with no rate and no cost", () => {
    const row = makeRow({
      provider: "openai",
      tracking_type: "duration_seconds",
      total_cost_microdollars: 0,
      total_duration_seconds: 100,
    });
    expect(estimateCostForRow(row, {})).toBeNull();
  });

  it("estimates cost from default rate for characters tracking", () => {
    const row = makeRow({
      provider: "elevenlabs",
      tracking_type: "characters",
      total_cost_microdollars: 0,
      total_tracking_amount: 2000,
    });
    // 2000 chars / 1000 * 0.18 USD * 1_000_000 = 360_000
    expect(estimateCostForRow(row, {})).toBe(360_000);
  });

  it("estimates cost from default rate for items tracking", () => {
    const row = makeRow({
      provider: "apollo",
      tracking_type: "items",
      total_cost_microdollars: 0,
      total_tracking_amount: 50,
    });
    // 50 * 0.02 * 1_000_000 = 1_000_000
    expect(estimateCostForRow(row, {})).toBe(1_000_000);
  });

  it("estimates cost from default rate for duration tracking", () => {
    const row = makeRow({
      provider: "e2b",
      tracking_type: "sandbox_seconds",
      total_cost_microdollars: 0,
      total_duration_seconds: 1_000_000,
    });
    // 1_000_000 * 0.000014 * 1_000_000 = 14_000_000
    expect(estimateCostForRow(row, {})).toBe(14_000_000);
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
    expect(trackingValue(row)).toBe("1.0K tokens");
  });

  it("returns formatted duration for sandbox_seconds", () => {
    const row = makeRow({
      tracking_type: "sandbox_seconds",
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

  it("returns formatted character count for characters tracking", () => {
    const row = makeRow({
      tracking_type: "characters",
      total_tracking_amount: 2500,
    });
    expect(trackingValue(row)).toBe("2.5K chars");
  });

  it("returns formatted item count for items tracking", () => {
    const row = makeRow({
      tracking_type: "items",
      total_tracking_amount: 1234,
    });
    expect(trackingValue(row)).toBe("1,234 items");
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

describe("toLocalInput", () => {
  it("returns empty string for empty input", () => {
    expect(toLocalInput("")).toBe("");
  });

  it("returns empty string for invalid ISO", () => {
    expect(toLocalInput("not-a-date")).toBe("");
  });

  it("converts UTC ISO to local datetime-local format", () => {
    const result = toLocalInput("2026-01-15T12:30:00Z");
    // Format should be YYYY-MM-DDTHH:mm
    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/);
  });
});

describe("toUtcIso", () => {
  it("returns empty string for empty input", () => {
    expect(toUtcIso("")).toBe("");
  });

  it("returns empty string for invalid local time", () => {
    expect(toUtcIso("not-a-date")).toBe("");
  });

  it("converts local datetime-local to ISO string", () => {
    const result = toUtcIso("2026-01-15T12:30");
    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
  });
});
