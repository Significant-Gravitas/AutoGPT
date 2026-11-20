"use client";

import { useEffect } from "react";
import { toast } from "sonner";

// One slot for both states, so "Connection restored" replaces "You're
// offline" instead of stacking on it. Pinned to the top: the default
// bottom-center sits over the chat composer and swallows clicks on it.
const NETWORK_TOAST = { id: "network-status", position: "top-center" } as const;
// Explicit because a replacement keeps the old toast's options: without it
// "Connection restored" inherits the offline toast's Infinity and never goes.
const RESTORED_DURATION_MS = 4_000;

export function useNetworkStatus() {
  useEffect(function monitorNetworkStatus() {
    function handleOnline() {
      toast.success("Connection restored", {
        ...NETWORK_TOAST,
        description: "You're back online",
        duration: RESTORED_DURATION_MS,
      });
    }

    function handleOffline() {
      toast.error("You're offline", {
        ...NETWORK_TOAST,
        description: "Check your internet connection",
        duration: Infinity,
      });
    }

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return function cleanup() {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);
}
