import type { CostLogRow } from "@/app/api/__generated__/models/costLogRow";
import type { ProviderCostSummary } from "@/app/api/__generated__/models/providerCostSummary";

const MICRODOLLARS_PER_USD = 1_000_000;

// Per-request cost estimates (USD) for providers billed per API call.
export const DEFAULT_COST_PER_RUN: Record<string, number> = {
  google_maps: 0.032, // $0.032/request - Google Maps Places API
  ideogram: 0.08, // $0.08/image - Ideogram standard generation
  nvidia: 0.0, // Free tier - NVIDIA NIM deepfake detection
  screenshotone: 0.01, // ~$0.01/screenshot - ScreenshotOne starter
  zerobounce: 0.008, // $0.008/validation - ZeroBounce
  mem0: 0.01, // ~$0.01/request - Mem0
  openweathermap: 0.0, // Free tier
  webshare_proxy: 0.0, // Flat subscription
  enrichlayer: 0.1, // ~$0.10/profile lookup
  jina: 0.0, // Free tier
};

export const DEFAULT_COST_PER_1K_TOKENS: Record<string, number> = {
  openai: 0.005,
  anthropic: 0.008,
  groq: 0.0003,
  ollama: 0.0,
  aiml_api: 0.005,
  llama_api: 0.003,
  v0: 0.005,
};

// Per-character rates (USD / 1K characters) for TTS providers.
export const DEFAULT_COST_PER_1K_CHARS: Record<string, number> = {
  unreal_speech: 0.008, // ~$8/1M chars on Starter
  elevenlabs: 0.18, // ~$0.18/1K chars on Starter
  d_id: 0.04, // ~$0.04/1K chars estimated
};

// Per-item rates (USD / item) for item-count billed APIs.
export const DEFAULT_COST_PER_ITEM: Record<string, number> = {
  google_maps: 0.017, // avg of $0.032 nearby + ~$0.015 detail enrich
  apollo: 0.02, // ~$0.02/contact on low-volume tiers
  smartlead: 0.001, // ~$0.001/lead added
};

// Per-second rates (USD / second) for duration-billed providers.
export const DEFAULT_COST_PER_SECOND: Record<string, number> = {
  e2b: 0.000014, // $0.000014/sec (2-core sandbox)
  fal: 0.0005, // varies by model, conservative
  replicate: 0.001, // varies by hardware
  revid: 0.01, // per-second of video
};

export function toDateOrUndefined(val?: string): Date | undefined {
  if (!val) return undefined;
  const d = new Date(val);
  return isNaN(d.getTime()) ? undefined : d;
}

export function formatMicrodollars(microdollars: number) {
  return `$${(microdollars / MICRODOLLARS_PER_USD).toFixed(4)}`;
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

// Unit label for each tracking type — what the rate input represents.
export function rateUnitLabel(trackingType: string | null | undefined): string {
  switch (trackingType) {
    case "tokens":
      return "$/1K tokens";
    case "characters":
      return "$/1K chars";
    case "items":
      return "$/item";
    case "sandbox_seconds":
    case "walltime_seconds":
      return "$/second";
    case "per_run":
      return "$/run";
    default:
      return "";
  }
}

// Default rate for a (provider, tracking_type) pair.
export function defaultRateFor(
  provider: string,
  trackingType: string | null | undefined,
): number | null {
  switch (trackingType) {
    case "tokens":
      return DEFAULT_COST_PER_1K_TOKENS[provider] ?? null;
    case "characters":
      return DEFAULT_COST_PER_1K_CHARS[provider] ?? null;
    case "items":
      return DEFAULT_COST_PER_ITEM[provider] ?? null;
    case "sandbox_seconds":
    case "walltime_seconds":
      return DEFAULT_COST_PER_SECOND[provider] ?? null;
    case "per_run":
      return DEFAULT_COST_PER_RUN[provider] ?? null;
    default:
      return null;
  }
}

// Overrides are keyed on `${provider}:${tracking_type}:${model}` since the
// same provider can have multiple rows with different billing models and models.
export function rateKey(
  provider: string,
  trackingType: string | null | undefined,
  model?: string | null,
): string {
  return `${provider}:${trackingType ?? "per_run"}:${model ?? ""}`;
}

export function estimateCostForRow(
  row: ProviderCostSummary,
  rateOverrides: Record<string, number>,
) {
  const tt = row.tracking_type || "per_run";

  // Providers that report USD directly: use known cost.
  if (tt === "cost_usd") return row.total_cost_microdollars;

  // Prefer the real USD the provider reported if any, but only for token paths
  // where OpenRouter piggybacks on the tokens row via x-total-cost.
  if (tt === "tokens" && row.total_cost_microdollars > 0) {
    return row.total_cost_microdollars;
  }

  const rate =
    rateOverrides[rateKey(row.provider, tt, row.model)] ??
    defaultRateFor(row.provider, tt);
  if (rate === null || rate === undefined) return null;

  // Compute the amount for this tracking type, then multiply by rate.
  let amount: number;
  switch (tt) {
    case "tokens": {
      // Anthropic cache tokens are billed at different rates:
      // - cache reads: 10% of base input rate
      // - cache writes: 125% of base input rate
      // - uncached input: 100% of base input rate
      const cacheRead = row.total_cache_read_tokens ?? 0;
      const cacheWrite = row.total_cache_creation_tokens ?? 0;
      if (cacheRead > 0 || cacheWrite > 0) {
        const uncachedInput = row.total_input_tokens;
        const output = row.total_output_tokens;
        const cost =
          (uncachedInput / 1000) * rate +
          (cacheRead / 1000) * rate * 0.1 +
          (cacheWrite / 1000) * rate * 1.25 +
          (output / 1000) * rate;
        return Math.round(cost * MICRODOLLARS_PER_USD);
      }
      // Rate is per-1K tokens.
      amount = (row.total_input_tokens + row.total_output_tokens) / 1000;
      break;
    }
    case "characters":
      // Rate is per-1K chars. trackingAmount aggregates char counts.
      amount = (row.total_tracking_amount || 0) / 1000;
      break;
    case "items":
      amount = row.total_tracking_amount || 0;
      break;
    case "sandbox_seconds":
    case "walltime_seconds":
      amount = row.total_duration_seconds || 0;
      break;
    case "per_run":
      amount = row.request_count;
      break;
    default:
      return row.total_cost_microdollars > 0
        ? row.total_cost_microdollars
        : null;
  }

  return Math.round(rate * amount * MICRODOLLARS_PER_USD);
}

export function trackingValue(row: ProviderCostSummary) {
  const tt = row.tracking_type || "per_run";
  if (tt === "cost_usd") return formatMicrodollars(row.total_cost_microdollars);
  if (tt === "tokens") {
    const tokens = row.total_input_tokens + row.total_output_tokens;
    const cacheRead = row.total_cache_read_tokens ?? 0;
    const cacheWrite = row.total_cache_creation_tokens ?? 0;
    if (cacheRead > 0 || cacheWrite > 0) {
      return `${formatTokens(tokens)} tokens (+${formatTokens(cacheRead)}r/${formatTokens(cacheWrite)}w cached)`;
    }
    return `${formatTokens(tokens)} tokens`;
  }
  if (tt === "sandbox_seconds" || tt === "walltime_seconds")
    return formatDuration(row.total_duration_seconds || 0);
  if (tt === "characters")
    return `${formatTokens(Math.round(row.total_tracking_amount || 0))} chars`;
  if (tt === "items")
    return `${Math.round(row.total_tracking_amount || 0).toLocaleString()} items`;
  return `${row.request_count.toLocaleString()} runs`;
}

// URL holds UTC ISO; datetime-local inputs need local "YYYY-MM-DDTHH:mm".
export function toLocalInput(iso: string) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// datetime-local emits naive local time; convert to UTC ISO so the
// backend filter window matches what the admin sees in their browser.
export function toUtcIso(local: string) {
  if (!local) return "";
  const d = new Date(local);
  return isNaN(d.getTime()) ? "" : d.toISOString();
}

const CSV_HEADERS = [
  "Time (UTC)",
  "User ID",
  "Email",
  "Block",
  "Provider",
  "Type",
  "Model",
  "Cost (USD)",
  "Input Tokens",
  "Output Tokens",
  "Cache Read Tokens",
  "Cache Creation Tokens",
  "Duration (s)",
  "Graph Exec ID",
  "Node Exec ID",
];

function csvEscape(val: unknown): string {
  const s = val == null ? "" : String(val);
  return `"${s.replace(/"/g, '""')}"`;
}

export function buildCostLogsCsv(logs: CostLogRow[]): string {
  const header = CSV_HEADERS.map(csvEscape).join(",");
  const rows = logs.map((log) =>
    [
      log.created_at,
      log.user_id,
      log.email,
      log.block_name,
      log.provider,
      log.tracking_type,
      log.model,
      log.cost_microdollars != null
        ? (log.cost_microdollars / 1_000_000).toFixed(8)
        : null,
      log.input_tokens,
      log.output_tokens,
      log.cache_read_tokens,
      log.cache_creation_tokens,
      log.duration,
      log.graph_exec_id,
      log.node_exec_id,
    ]
      .map(csvEscape)
      .join(","),
  );
  return [header, ...rows].join("\r\n");
}
