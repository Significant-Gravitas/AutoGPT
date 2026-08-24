export type CompactionPhase = "summarizing" | "rebuilding" | "done";

export interface CompactionStats {
  tokensBefore?: number;
  tokensAfter?: number;
  messagesBefore?: number;
  messagesAfter?: number;
}

// Each phase approaches its own ceiling exponentially, so the bar can never
// land before the work does. Entering a new phase raises the ceiling, which
// reads as real progress rather than a stalled bar.
export const PHASE_CURVE: Record<
  CompactionPhase,
  { cap: number; tauMs: number }
> = {
  summarizing: { cap: 0.55, tauMs: 15_000 },
  rebuilding: { cap: 0.92, tauMs: 20_000 },
  done: { cap: 1, tauMs: 420 },
};

export const FINISH_MS = 420;
export const INITIAL_PROGRESS = 0.02;

const TAU_FLOOR_MS = 12_000;
const TAU_CEILING_MS = 45_000;
const TAU_MS_PER_TOKEN = 1_000 / 10_000;

export function tauForTokens(tokensBefore: number | undefined): number {
  const scaled = (tokensBefore ?? 0) * TAU_MS_PER_TOKEN;
  return Math.min(TAU_CEILING_MS, Math.max(TAU_FLOOR_MS, scaled));
}

export function phaseProgress(
  base: number,
  cap: number,
  elapsedMs: number,
  tauMs: number,
): number {
  // Floor the decay term to keep the asymptote strictly below cap in float64.
  const decay = Math.max(Math.exp(-elapsedMs / tauMs), 1e-9);
  return base + (cap - base) * (1 - decay);
}

export function finishProgress(base: number, elapsedMs: number): number {
  const eased = 1 - Math.exp((-4 * elapsedMs) / FINISH_MS);
  return Math.min(1, base + (1 - base) * eased);
}

const LEGACY_SUMMARY =
  "Earlier messages were summarized to fit within context limits.";

function readStats(source: Record<string, unknown>): CompactionStats {
  const stats: CompactionStats = {};
  const keys = [
    "tokensBefore",
    "tokensAfter",
    "messagesBefore",
    "messagesAfter",
  ] as const;
  for (const key of keys) {
    const value = source[key];
    if (typeof value === "number" && Number.isFinite(value)) stats[key] = value;
  }
  return stats;
}

export function parseCompactionOutput(output: unknown): {
  summary: string;
  stats: CompactionStats;
} {
  let value: unknown = output;
  if (typeof value === "string") {
    const text = value;
    try {
      value = JSON.parse(text);
    } catch {
      return { summary: text, stats: {} };
    }
  }
  if (typeof value !== "object" || value === null) {
    return { summary: LEGACY_SUMMARY, stats: {} };
  }
  const record = value as Record<string, unknown>;
  const summary =
    typeof record.summary === "string" ? record.summary : LEGACY_SUMMARY;
  return { summary, stats: readStats(record) };
}

export function formatTokens(n: number): string {
  if (n < 1_000) return String(n);
  if (n < 10_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${Math.round(n / 1_000)}K`;
}

export function compactionLabel(
  phase: CompactionPhase | null,
  stats: CompactionStats,
): string {
  if (phase === "summarizing") return "Condensing our conversation…";
  if (phase === "rebuilding") return "Reloading context…";

  const parts: string[] = [];
  if (phase === "done" && stats.messagesBefore !== undefined) {
    parts.push(`Condensed ${stats.messagesBefore} messages`);
  } else {
    parts.push("Condensed the conversation");
  }
  if (stats.tokensBefore !== undefined && stats.tokensAfter !== undefined) {
    parts.push(
      `${formatTokens(stats.tokensBefore)} → ${formatTokens(stats.tokensAfter)} tokens`,
    );
  }
  if (parts.length === 1 && phase !== "done") {
    return "Condensed the conversation to keep going";
  }
  return parts.join(" · ");
}
