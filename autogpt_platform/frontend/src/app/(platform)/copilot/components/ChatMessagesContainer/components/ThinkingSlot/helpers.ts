import type { TurnActivity } from "../../thinkingActivity";

/** Text that has not grown for this long has stalled: the indicator comes
 *  back while the model works out what to do next. */
export const TEXT_IDLE_MS = 1000;
/** A gap between steps shorter than this draws nothing: the next tool row
 *  usually lands well inside it. */
export const STEP_GAP_MS = 250;
/** Once shown, the indicator stays at least this long, so it never blinks. */
export const MIN_VISIBLE_MS = 400;
/** The fade out; shorter than the 200 ms fade in. */
export const EXIT_MS = 150;

export const now = () => Date.now();

export interface Wait {
  /** Changes whenever the wait starts over; null when nothing is awaited. */
  key: string | null;
  /** How long the wait must hold before the indicator shows. */
  delayMs: number;
  /** An indicator already showing stays up through this wait. */
  keepsShowing: boolean;
}

export function getWait(activity: TurnActivity, enabled: boolean): Wait {
  if (!enabled || activity.kind === "ended" || activity.kind === "working") {
    return { key: null, delayMs: 0, keepsShowing: false };
  }
  if (activity.kind === "writing") {
    return {
      key: `writing:${activity.signature}`,
      delayMs: TEXT_IDLE_MS,
      keepsShowing: false,
    };
  }
  return {
    key: "quiet",
    delayMs: activity.midTurn ? STEP_GAP_MS : 0,
    keepsShowing: true,
  };
}
