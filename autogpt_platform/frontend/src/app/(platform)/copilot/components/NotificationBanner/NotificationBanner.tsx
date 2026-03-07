"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Key, storage } from "@/services/storage/local-storage";
import { BellRinging, X } from "@phosphor-icons/react";
import { useState } from "react";

export function NotificationBanner() {
  const [dismissed, setDismissed] = useState(
    () => storage.get(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED) === "true",
  );
  const [permission, setPermission] = useState(() =>
    typeof Notification !== "undefined" ? Notification.permission : "denied",
  );

  // Don't show if notifications aren't supported, already decided, or dismissed
  if (
    typeof Notification === "undefined" ||
    permission !== "default" ||
    dismissed
  ) {
    return null;
  }

  function handleEnable() {
    Notification.requestPermission().then((result) => {
      setPermission(result);
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
        Enable browser notifications to know when Otto finishes working, even
        when you switch tabs.
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
