"use client";

import { useFlags } from "launchdarkly-react-client-sdk";
import type { FlagSourceResult } from "../flag-source";

export function useLaunchDarklyFlag(key: string): FlagSourceResult {
  const currentFlags = useFlags<Record<string, unknown>>();

  return { value: currentFlags[key], resolved: key in currentFlags };
}
