"use client";

import { useSyncExternalStore } from "react";

import { createWallClock, useWallClock } from "@/hooks/useNowSecond";

const minuteClock = createWallClock(60_000);

function subscribeVisibility(listener: () => void) {
  document.addEventListener("visibilitychange", listener);
  return () => document.removeEventListener("visibilitychange", listener);
}

function isVisible() {
  return document.visibilityState !== "hidden";
}

/** Epoch milliseconds of the current minute: ticks once a minute while the
 *  tab is visible, rests while it is hidden, and catches up when it shows. */
export function useMinuteClock() {
  const visible = useSyncExternalStore(
    subscribeVisibility,
    isVisible,
    () => true,
  );
  return useWallClock(minuteClock, visible);
}
