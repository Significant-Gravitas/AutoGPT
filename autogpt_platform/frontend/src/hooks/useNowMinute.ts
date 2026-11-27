import { createWallClock, useWallClock } from "./useNowSecond";

const minuteClock = createWallClock(60_000);

/**
 * Epoch milliseconds of the current wall-clock minute. Re-renders once a
 * minute, in step with every other `useNowMinute` caller, while `enabled`.
 */
export function useNowMinute(enabled = true) {
  return useWallClock(minuteClock, enabled);
}
