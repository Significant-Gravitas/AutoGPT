import type { ProviderCostSummary } from "@/app/api/__generated__/models/providerCostSummary";

export const DEFAULT_COST_PER_RUN: Record<string, number> = {
  google_maps: 0.032,
  ideogram: 0.08,
  nvidia: 0.0,
  screenshotone: 0.01,
  zerobounce: 0.008,
  mem0: 0.01,
  openweathermap: 0.0,
  webshare_proxy: 0.0,
};

export const DEFAULT_COST_PER_1K_TOKENS: Record<string, number> = {
  openai: 0.005,
  anthropic: 0.008,
  groq: 0.0003,
  ollama: 0.0,
};

export function toDateOrUndefined(val?: string): Date | undefined {
  if (!val) return undefined;
  const d = new Date(val);
  return isNaN(d.getTime()) ? undefined : d;
}

export function formatMicrodollars(microdollars: number) {
  return `$${(microdollars / 1_000_000).toFixed(4)}`;
}

export function formatTokens(tokens: number) {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(1)}K`;
  return tokens.toString();
}

export function formatDuration(seconds: number) {
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)}h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  return `${seconds.toFixed(1)}s`;
}

export function estimateCostForRow(
  row: ProviderCostSummary,
  costPerRunOverrides: Record<string, number>,
) {
  const tt = row.tracking_type || "per_run";
  if (tt === "cost_usd") return row.total_cost_microdollars;
  if (tt === "tokens") {
    if (row.total_cost_microdollars > 0) return row.total_cost_microdollars;
    const rate = DEFAULT_COST_PER_1K_TOKENS[row.provider] ?? null;
    if (rate !== null) {
      const totalTokens = row.total_input_tokens + row.total_output_tokens;
      return Math.round((totalTokens / 1000) * rate * 1_000_000);
    }
    return null;
  }
  if (tt === "per_run") {
    const rate =
      costPerRunOverrides[row.provider] ??
      DEFAULT_COST_PER_RUN[row.provider] ??
      null;
    if (rate !== null) return Math.round(rate * row.request_count * 1_000_000);
    return null;
  }
  return row.total_cost_microdollars > 0 ? row.total_cost_microdollars : null;
}

export function trackingValue(row: ProviderCostSummary) {
  const tt = row.tracking_type || "per_run";
  if (tt === "cost_usd") return formatMicrodollars(row.total_cost_microdollars);
  if (tt === "tokens")
    return formatTokens(row.total_input_tokens + row.total_output_tokens);
  if (
    tt === "duration_seconds" ||
    tt === "sandbox_seconds" ||
    tt === "walltime_seconds"
  )
    return formatDuration(row.total_duration_seconds || 0);
  if (tt === "characters")
    return formatTokens(row.total_input_tokens + row.total_output_tokens);
  return row.request_count.toLocaleString() + " runs";
}
