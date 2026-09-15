// The wire contract: the backend only ever emits these two phases. There is
// no "done" — a finished compaction settles the row instead.
export type CompactionPhase = "summarizing" | "rebuilding";

/**
 * AI SDK v5 wire type of the transient progress part the backend streams
 * while a compaction runs (`ResponseType.COMPACTION`). Every filter that has
 * to see past it references this constant — spelling the string by hand in
 * one more place is how a bookkeeping part starts rendering as content.
 */
export const COMPACTION_DATA_PART_TYPE = "data-compaction";

export interface CompactionStats {
  tokensBefore?: number;
  tokensAfter?: number;
  messagesBefore?: number;
  messagesAfter?: number;
  // The backend could not condense the history and dropped it instead. A
  // settled row carrying this reports the reset — never "Condensed…".
  dropped?: true;
}

export const DROPPED_LABEL =
  "Started a fresh context — earlier messages were dropped";

// Each phase approaches its own ceiling exponentially, so the bar can never
// land before the work does. Entering a new phase raises the ceiling, which
// reads as real progress rather than a stalled bar.
export const PHASE_CURVE: Record<
  CompactionPhase,
  { cap: number; tauMs: number }
> = {
  summarizing: { cap: 0.55, tauMs: 15_000 },
  rebuilding: { cap: 0.92, tauMs: 20_000 },
};

export const INITIAL_PROGRESS = 0.02;
// Within half a percent of the ceiling the rounded bar can no longer move,
// so the rAF loop parks itself instead of spinning at 60fps forever.
export const SETTLE_EPSILON = 0.005;

// How often a curve parked at its ceiling re-checks for a new phase. Slow
// enough that a stalled rebuild costs nothing, fast enough that the handover
// to the next phase reads as continuous.
export const PARKED_POLL_MS = 1_000;

// Under `prefers-reduced-motion` the bar advances in whole steps instead of
// creeping a pixel at a time. The progress is the feature — a compaction can
// run for minutes and the user deserves to know it is moving — but the crawl
// is the motion, so we keep the information and drop the animation.
export const REDUCED_MOTION_STEP_PERCENT = 10;

/**
 * Width the bar actually paints. Always floors to the step under reduced
 * motion: rounding up would claim progress the curve has not made.
 */
export function barPercent(progress: number, prefersReducedMotion: boolean) {
  const percent = Math.round(progress * 100);
  if (!prefersReducedMotion) return percent;
  return (
    Math.floor(percent / REDUCED_MOTION_STEP_PERCENT) *
    REDUCED_MOTION_STEP_PERCENT
  );
}

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

export function readCompactionStats(
  source: Record<string, unknown>,
): CompactionStats {
  const stats: CompactionStats = {};
  const keys = [
    "tokensBefore",
    "tokensAfter",
    "messagesBefore",
    "messagesAfter",
  ] as const;
  for (const key of keys) {
    const value = source[key];
    // Counts are positive integers by construction — anything else (0, a
    // fraction, NaN) is a measurement bug and must not reach the copy.
    if (typeof value === "number" && Number.isInteger(value) && value > 0) {
      stats[key] = value;
    }
  }
  if (source.dropped === true) stats.dropped = true;
  return stats;
}

/**
 * Stats from a settled row's output. The payload's `summary` prose is
 * transcript-level detail the card deliberately does not surface — the
 * settled copy is `compactionLabel`'s verified claim, so legacy
 * plain-sentence outputs simply yield no stats.
 */
export function parseCompactionOutput(output: unknown): CompactionStats {
  let value: unknown = output;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return {};
    }
  }
  if (typeof value !== "object" || value === null) return {};
  return readCompactionStats(value as Record<string, unknown>);
}

function formatWithUnit(value: number, unit: string): string {
  const rounded = value < 10 ? Math.round(value * 10) / 10 : Math.round(value);
  const text = Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  return `${text}${unit}`;
}

export function formatTokens(n: number): string {
  if (n < 1_000) return String(n);
  // 999_500 rounds up to 1M — switching units there keeps the scale
  // continuous ("999K" → "1M", never "1000K").
  if (n < 999_500) return formatWithUnit(n / 1_000, "K");
  return formatWithUnit(n / 1_000_000, "M");
}

export function compactionLabel(
  phase: CompactionPhase | null,
  stats: CompactionStats,
): string {
  if (phase === "summarizing") return "Condensing our conversation…";
  if (phase === "rebuilding") return "Reloading context…";
  if (stats.dropped) return DROPPED_LABEL;

  // A number only earns its place when both ends of the measurement exist
  // and the "after" actually shrank — an equal or inverted pair (e.g. 60
  // messages summarized in place, none removed) reads as a broken claim.
  const condensedMessages =
    stats.messagesBefore !== undefined &&
    stats.messagesAfter !== undefined &&
    stats.messagesAfter < stats.messagesBefore;
  const headline = condensedMessages
    ? `Condensed ${stats.messagesBefore} messages`
    : "Condensed the conversation";
  const tokens =
    stats.tokensBefore !== undefined &&
    stats.tokensAfter !== undefined &&
    stats.tokensAfter < stats.tokensBefore
      ? `${formatTokens(stats.tokensBefore)} → ${formatTokens(stats.tokensAfter)} tokens`
      : null;
  if (tokens) return `${headline} · ${tokens}`;
  return condensedMessages
    ? headline
    : "Condensed the conversation to keep going";
}
