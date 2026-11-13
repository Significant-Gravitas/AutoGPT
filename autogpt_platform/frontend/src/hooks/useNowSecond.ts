import { useSyncExternalStore } from "react";

interface WallClock {
  subscribe(listener: () => void): () => void;
  getSnapshot(): number;
}

/**
 * One shared clock that ticks on the wall-clock `periodMs` boundary. A single
 * `setTimeout` serves every subscriber, so all of them see the same value in
 * the same tick. The timer starts with the first subscriber and stops with
 * the last. Values are epoch milliseconds floored to the period, so they only
 * change once per tick.
 */
export function createWallClock(periodMs: number): WallClock {
  const listeners = new Set<() => void>();
  let timer: ReturnType<typeof setTimeout> | null = null;
  let current = 0;

  function floorNow() {
    return Math.floor(Date.now() / periodMs) * periodMs;
  }

  function schedule() {
    timer = setTimeout(tick, periodMs - (Date.now() % periodMs));
  }

  function tick() {
    schedule();
    const next = floorNow();
    if (next === current) return;
    current = next;
    listeners.forEach((listener) => listener());
  }

  return {
    subscribe(listener) {
      listeners.add(listener);
      if (timer === null) {
        current = floorNow();
        schedule();
      }
      return () => {
        listeners.delete(listener);
        if (listeners.size > 0 || timer === null) return;
        clearTimeout(timer);
        timer = null;
      };
    },
    getSnapshot() {
      return timer === null ? floorNow() : current;
    },
  };
}

function subscribeNever() {
  return () => {};
}

export function useWallClock(clock: WallClock, enabled: boolean) {
  return useSyncExternalStore(
    enabled ? clock.subscribe : subscribeNever,
    clock.getSnapshot,
    clock.getSnapshot,
  );
}

const secondClock = createWallClock(1_000);

/**
 * Epoch milliseconds of the current wall-clock second. Re-renders once a
 * second, in step with every other `useNowSecond` caller, while `enabled`.
 */
export function useNowSecond(enabled = true) {
  return useWallClock(secondClock, enabled);
}
