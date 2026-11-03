/**
 * Why a turn stopped at a limit instead of failing.
 *
 * A turn that runs out of budget or steps now lands on purpose: the agent
 * writes a checkpoint and the chat shows a Paused card offering the ways to
 * pick the task back up. The payload arrives live as a `data-turn-paused`
 * part and is stored on a marker row, so a reload rebuilds the same card. One
 * parser for both, the way `providerFailure.ts` reads `data-provider-failure`.
 */

export const TURN_PAUSED_PART_TYPE = "data-turn-paused";

/** The key the backend stores the pause under on a marker row. */
export const TURN_PAUSE_KEY = "turn_paused";

export type TurnPauseReason = "daily" | "weekly" | "trial" | "task" | "steps";

export interface TurnPause {
  reason: TurnPauseReason;
  /** When the limiting window refills; set for daily and weekly pauses. */
  resets_at: string | null;
  /** Model calls the turn made; set for step check-ins. */
  steps: number | null;
  /** False when the turn was stopped before it could write its checkpoint. */
  checkpoint_saved: boolean;
  /** False when the turn was refused before it ran at all. */
  started: boolean;
}

const REASONS = new Set<string>(["daily", "weekly", "trial", "task", "steps"]);

/** The pause, from the stream part's data or the marker row's metadata. */
export function parseTurnPause(value: unknown): TurnPause | null {
  if (!value || typeof value !== "object") return null;
  const data = value as Record<string, unknown>;
  if (typeof data.reason !== "string" || !REASONS.has(data.reason)) {
    return null;
  }
  return {
    reason: data.reason as TurnPauseReason,
    resets_at: isoOrNull(data.resets_at),
    steps:
      typeof data.steps === "number" && Number.isFinite(data.steps)
        ? data.steps
        : null,
    // The backend model defaults both to true.
    checkpoint_saved: data.checkpoint_saved !== false,
    started: data.started !== false,
  };
}

export function parseTurnPausePart(part: {
  type: string;
  data?: unknown;
}): TurnPause | null {
  if (part.type !== TURN_PAUSED_PART_TYPE) return null;
  return parseTurnPause(part.data);
}

/** The pause a stored row records, if the row is a pause marker. */
export function turnPauseOfMetadata(metadata: unknown): TurnPause | null {
  if (!metadata || typeof metadata !== "object") return null;
  return parseTurnPause((metadata as Record<string, unknown>)[TURN_PAUSE_KEY]);
}

// Kept in step with `_SUMMARIES` / `_NOT_STARTED` in the backend's
// `copilot/turn_pause.py`: the marker row's text is part of the checkpoint
// digest, so the stream fold has to write the same words the server does.
const SUMMARIES: Record<TurnPauseReason, string> = {
  daily: "Paused: this task is out of today's budget.",
  weekly: "Paused: this task is out of this week's budget.",
  trial: "Paused: the trial allowance is used up.",
  task: "Paused: this task reached its size limit.",
  steps: "Paused: checking in after a long run.",
};

const NOT_STARTED: Record<TurnPauseReason, string> = {
  daily: "Not started: out of today's budget.",
  weekly: "Not started: out of this week's budget.",
  trial: "Not started: the trial allowance is used up.",
  task: "Not started.",
  steps: "Not started.",
};

/** The system note the marker row carries for clients that predate the card. */
export function turnPauseSummary(pause: TurnPause): string {
  return pause.started ? SUMMARIES[pause.reason] : NOT_STARTED[pause.reason];
}

// The API client turns ISO strings into Dates on the way in, so a stored row
// can carry either.
function isoOrNull(value: unknown): string | null {
  if (typeof value === "string") return value;
  if (value instanceof Date && Number.isFinite(value.getTime())) {
    return value.toISOString();
  }
  return null;
}
