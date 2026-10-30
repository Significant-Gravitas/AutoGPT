import {
  readPermission,
  subscribeToPermission,
} from "@/components/layout/NotificationSettings/helpers";
import { useNotificationSettings } from "@/components/layout/NotificationSettings/useNotificationSettings";
import { useSyncExternalStore } from "react";
import { useCopilotUIStore } from "../../store";

function hiddenOnServer() {
  return "unsupported" as const;
}

export function useNotificationBanner() {
  const { isNotificationsEnabled, toggleNotifications } =
    useNotificationSettings();
  const isDismissed = useCopilotUIStore((s) => s.isNotificationBannerDismissed);
  const dismissNotificationBanner = useCopilotUIStore(
    (s) => s.dismissNotificationBanner,
  );
  // Live, so granting permission from settings in another tab hides this one
  // when the user comes back. Hidden on the server and while hydrating.
  const permission = useSyncExternalStore(
    subscribeToPermission,
    readPermission,
    hiddenOnServer,
  );

  // A granted permission with our switch off is a browser that allowed us
  // before, e.g. after signing out and back in: one click turns it back on.
  const canAsk = permission === "default" || permission === "granted";
  const isVisible = canAsk && !isDismissed && !isNotificationsEnabled;

  function handleTurnOn() {
    void toggleNotifications();
  }

  return {
    isVisible,
    handleTurnOn,
    handleDismiss: dismissNotificationBanner,
  };
}
