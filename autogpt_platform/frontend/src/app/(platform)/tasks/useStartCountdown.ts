import {
  createWallClock,
  useNowSecond,
  useWallClock,
} from "@/hooks/useNowSecond";
import { formatStartTime, formatStartsIn, startTimeOf } from "./schedule";

const halfMinuteClock = createWallClock(30_000);

/** Inside this window before the start the countdown ticks every second;
 *  wider than a minute because the half-minute clock can lag by 30 s. */
const FINE_MS = 90_000;
/** A start this far past stops the fast tick: the task has moved on. */
const PAST_MS = 30_000;

/** A live "Starts in …" for a start time, with the exact time beside it. It
 *  ticks once a second in the last minute and every 30 s before that, and is
 *  worked out from the start time itself, so a refresh never drifts. Null
 *  and idle, with no timer, when there is no start time. */
export function useStartCountdown(startAt: Date | string | null | undefined) {
  const target = startTimeOf(startAt);
  const isActive = Number.isFinite(target);
  const coarse = useWallClock(halfMinuteClock, isActive);
  const isNear =
    isActive && target - coarse < FINE_MS && target - coarse > -PAST_MS;
  const fine = useNowSecond(isNear);

  if (!isActive) return null;
  const now = isNear ? Math.max(fine, coarse) : coarse;
  return {
    label: formatStartsIn(target, now),
    exact: formatStartTime(target, now),
    isDue: target <= now,
  };
}
