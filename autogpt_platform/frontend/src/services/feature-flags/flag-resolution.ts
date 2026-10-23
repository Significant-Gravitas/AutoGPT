"use client";

import { createContext, useContext } from "react";

// Whether the configured vendor has been given the chance to answer. The
// LaunchDarkly provider holds initialisation until the session resolves, and
// during that wait the fallback timeout in ``useFlagStatus`` must not count
// down: timing out there serves the default, and a route gated on the flag
// would 404 a user who has it on. Defaults to true so that with no deferring
// provider above (PostHog, flags disabled, tests) the timeout runs from mount.
export const FlagResolutionContext = createContext(true);

export function useFlagResolutionStarted() {
  return useContext(FlagResolutionContext);
}
