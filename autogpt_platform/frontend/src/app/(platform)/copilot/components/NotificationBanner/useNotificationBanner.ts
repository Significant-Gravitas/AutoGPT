import {
  readPermission,
  subscribeToPermission,
} from "@/components/layout/NotificationSettings/helpers";
import { useNotificationSettings } from "@/components/layout/NotificationSettings/useNotificationSettings";
import { Key, storage } from "@/services/storage/local-storage";
import { useReducer, useSyncExternalStore } from "react";

function hiddenOnServer() {
  return "unsupported" as const;
}

export function useNotificationBanner() {
  const { isNotificationsEnabled, toggleNotifications } =
    useNotificationSettings();

  const [, rerender] = useReducer((count: number) => count + 1, 0);
  // Live, so granting permission from settings in another tab hides this one
  // when the user comes back. Hidden on the server and while hydrating.
  const permission = useSyncExternalStore(
    subscribeToPermission,
    readPermission,
    hiddenOnServer,
  );

  // Storage is read on every render: signing out clears the dismissal, and
  // turning notifications on in another tab only reaches this one through
  // storage, picked up on the render the permission check above triggers.
  // A granted permission with our switch off is a browser that allowed us
  // before, e.g. after signing out and back in: one click turns it back on.
  const dismissed =
    storage.get(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED) === "true";
  const canAsk = permission === "default" || permission === "granted";
  const isOn =
    isNotificationsEnabled ||
    (permission === "granted" &&
      storage.get(Key.COPILOT_NOTIFICATIONS_ENABLED) === "true");
  const isVisible = canAsk && !dismissed && !isOn;

  function handleTurnOn() {
    void toggleNotifications();
  }

  function handleDismiss() {
    storage.set(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED, "true");
    rerender();
  }

  return { isVisible, handleTurnOn, handleDismiss };
}
