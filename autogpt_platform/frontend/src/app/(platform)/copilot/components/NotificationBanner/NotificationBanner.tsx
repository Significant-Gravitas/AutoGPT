"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Key, storage } from "@/services/storage/local-storage";
import { BellRingingIcon, XIcon } from "@phosphor-icons/react";
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
    <Alert variant="warning" icon={BellRingingIcon} aria-live="polite">
      <div className="flex flex-wrap items-center gap-3">
        <AlertDescription className="min-w-[12rem] flex-1">
          Enable browser notifications to know when AutoPilot finishes working,
          even when you switch tabs.
        </AlertDescription>
        <Button variant="primary" size="small" onClick={handleEnable}>
          Enable
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={handleDismiss}
          aria-label="Dismiss"
          title="Dismiss"
          className="hover:border-[#FFE4BF] hover:bg-[#FFE4BF]"
        >
          <XIcon className="h-4 w-4" />
        </Button>
      </div>
    </Alert>
  );
}
