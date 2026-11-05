"use client";

import { useServerFlag } from "./backends/server";

export interface FlagSourceResult {
  value: unknown;
  // Whether the flag service actually answered, as opposed to not having
  // answered yet. Callers that gate a route must not treat "no answer" as "off".
  resolved: boolean;
}

// The one seam every flag read passes through, and the only thing a test has
// to mock to control flags.
export function useFlagSource(key: string): FlagSourceResult {
  return useServerFlag(key);
}
