"use client";

import { useEffect, useState } from "react";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { toast } from "@/components/molecules/Toast/use-toast";

function readPermission(): NotificationPermission | "unsupported" {
  if (typeof Notification === "undefined") return "unsupported";
  return Notification.permission;
}

/**
 * The one place that knows how the browser's three-state permission maps onto
 * our single app-level switch.
 *
 * We can never revoke a granted permission — "off" only ever means our own
 * flag. Once the user has denied at the browser level, the switch is inert
 * and the surface has to send them to their browser's site settings.
 */
export function useNotificationSettings() {
  const isNotificationsEnabled = useCopilotUIStore(
    (s) => s.isNotificationsEnabled,
  );
  const setNotificationsEnabled = useCopilotUIStore(
    (s) => s.setNotificationsEnabled,
  );
  const isSoundEnabled = useCopilotUIStore((s) => s.isSoundEnabled);
  const toggleSound = useCopilotUIStore((s) => s.toggleSound);

  // `Notification.permission` doesn't exist during SSR and reading it during
  // render would desync hydration, so it lands after mount.
  const [permission, setPermission] = useState<
    NotificationPermission | "unsupported"
  >("default");
  useEffect(() => setPermission(readPermission()), []);

  const isSupported = permission !== "unsupported";
  const isBlocked = permission === "denied";

  async function toggleNotifications() {
    if (isNotificationsEnabled) {
      setNotificationsEnabled(false);
      return;
    }

    if (typeof Notification === "undefined") {
      toast({
        title: "Notifications not supported",
        description: "Your browser does not support notifications.",
        variant: "destructive",
      });
      return;
    }

    const result = await Notification.requestPermission();
    setPermission(result);

    if (result === "granted") {
      setNotificationsEnabled(true);
      return;
    }

    toast({
      title: "Notifications blocked",
      description:
        "Please allow notifications in your browser settings to enable this feature.",
      variant: "destructive",
    });
  }

  return {
    isNotificationsEnabled,
    isSoundEnabled,
    isSupported,
    isBlocked,
    permission,
    toggleNotifications,
    toggleSound,
  };
}
