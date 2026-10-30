import {
  readPermission,
  subscribeToPermission,
} from "@/components/layout/NotificationSettings/helpers";
import { useNotificationSettings } from "@/components/layout/NotificationSettings/useNotificationSettings";
import { toast } from "@/components/molecules/Toast/use-toast";
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
  // before our switch was ever turned on: one click turns it on.
  const canAsk = permission === "default" || permission === "granted";
  const isVisible = canAsk && !isDismissed && !isNotificationsEnabled;

  // The banner disappears with the focused button, so say it worked.
  async function handleTurnOn() {
    await toggleNotifications();
    if (useCopilotUIStore.getState().isNotificationsEnabled) {
      toast({
        title: "Notifications are on",
        description: "We'll let you know when Otto or your experts finish.",
      });
    }
  }

  return {
    isVisible,
    handleTurnOn,
    handleDismiss: dismissNotificationBanner,
  };
}
