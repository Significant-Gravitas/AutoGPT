"use client";

import {
  getPersistedNotificationsEnabled,
  useCopilotUIStore,
} from "@/app/(platform)/copilot/store";
import { Key } from "@/services/storage/local-storage";
import { useEffect } from "react";

/**
 * Keeps this tab's notifications switch in step with the other tabs. The
 * switch drives the shared push subscription and the in-page fallback, so a
 * tab still holding an old value would either show notifications after
 * another tab switched them off, or remove the subscription after another tab
 * switched them on.
 */
export function useSyncNotificationsEnabled() {
  useEffect(() => {
    function handleStorage(event: StorageEvent) {
      if (
        event.key !== null &&
        event.key !== Key.COPILOT_NOTIFICATIONS_ENABLED
      ) {
        return;
      }
      useCopilotUIStore.setState({
        isNotificationsEnabled: getPersistedNotificationsEnabled(),
      });
    }

    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);
}
