"use client";

import {
  getPersistedNotificationBannerDismissed,
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
 * switched them on. The chat's turn-on banner follows too, so an "off" chosen
 * in another tab doesn't bring it back here.
 */
export function useSyncNotificationsEnabled() {
  useEffect(() => {
    function syncFromStorage() {
      useCopilotUIStore.setState({
        isNotificationsEnabled: getPersistedNotificationsEnabled(),
        isNotificationBannerDismissed:
          getPersistedNotificationBannerDismissed(),
      });
    }

    function handleStorage(event: StorageEvent) {
      if (
        event.key !== null &&
        event.key !== Key.COPILOT_NOTIFICATIONS_ENABLED &&
        event.key !== Key.COPILOT_NOTIFICATION_BANNER_DISMISSED
      ) {
        return;
      }
      syncFromStorage();
    }

    // The store read the setting when it loaded; another tab may have
    // changed it before this listener existed.
    syncFromStorage();
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);
}
