"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Key, storage } from "@/services/storage/local-storage";
import { BellRinging, X } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { useCopilotUIStore } from "../../store";

export function NotificationBanner() {
  const { setNotificationsEnabled, isNotificationsEnabled } =
    useCopilotUIStore();

  const [dismissed, setDismissed] = useState(
    () => storage.get(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED) === "true",
  );
  const [permission, setPermission] = useState(() =>
    typeof Notification !== "undefined" ? Notification.permission : "denied",
  );

  // Re-read dismissed flag when notifications are toggled off (e.g. clearCopilotLocalData)
  useEffect(() => {
    if (!isNotificationsEnabled) {
      setDismissed(
        storage.get(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED) === "true",
      );
    }
  }, [isNotificationsEnabled]);

  // Don't show if notifications aren't supported, already decided, dismissed, or already enabled
  if (
    typeof Notification === "undefined" ||
    permission !== "default" ||
    dismissed ||
    isNotificationsEnabled
  ) {
    return null;
  }

  function handleEnable() {
    Notification.requestPermission().then((result) => {
      setPermission(result);
      if (result === "granted") {
        setNotificationsEnabled(true);
        handleDismiss();
      }
    });
  }

  function handleDismiss() {
    storage.set(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED, "true");
    setDismissed(true);
  }

  return (
    <div className="flex items-center gap-3 border-b border-amber-200 bg-amber-50 px-4 py-2.5">
      <BellRinging className="h-5 w-5 shrink-0 text-amber-600" weight="fill" />
      <Text variant="body" className="flex-1 text-sm text-amber-800">
        Enable browser notifications to know when AutoPilot finishes working,
        even when you switch tabs.
      </Text>
      <Button variant="primary" size="small" onClick={handleEnable}>
        Enable
      </Button>
      <button
        onClick={handleDismiss}
        className="rounded p-1 text-amber-400 transition-colors hover:text-amber-600"
        aria-label="Dismiss"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}
