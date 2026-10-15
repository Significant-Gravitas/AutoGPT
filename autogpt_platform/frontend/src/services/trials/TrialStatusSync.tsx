"use client";

import { useTrialStatus } from "./useTrialStatus";

export function TrialStatusSync() {
  useTrialStatus({ poll: true });
  return null;
}
