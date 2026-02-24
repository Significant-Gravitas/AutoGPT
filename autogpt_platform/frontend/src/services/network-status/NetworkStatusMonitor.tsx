"use client";

import { useNetworkStatus } from "./useNetworkStatus";

export function NetworkStatusMonitor() {
  useNetworkStatus();
  return null;
}
