"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Key, storage } from "@/services/storage/local-storage";
import {
  type MouseEvent,
  useEffect,
  useState,
  useSyncExternalStore,
} from "react";
import { useCopilotUIStore } from "../../store";
import {
  readPermission,
  subscribeToPermission,
} from "@/components/layout/NotificationSettings/helpers";
import { isPlainLeftClick } from "./helpers";
import { BellRingIcon, Cancel01Icon } from "@hugeicons/core-free-icons";
import { createIconComponent, Icon } from "@/components/atoms/Icon/Icon";

// Alert's `icon` prop takes a component (its defaults come from lucide).
const BellRing = createIconComponent(BellRingIcon);

function hiddenOnServer() {
  return "unsupported" as const;
}

export function NotificationBanner() {
  const isNotificationsEnabled = useCopilotUIStore(
    (state) => state.isNotificationsEnabled,
  );

  const [dismissed, setDismissed] = useState(
    () => storage.get(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED) === "true",
  );
  // Live, so granting permission from settings in another tab hides this one
  // when the user comes back. Hidden on the server and while hydrating.
  const permission = useSyncExternalStore(
    subscribeToPermission,
    readPermission,
    hiddenOnServer,
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

  function persistDismissed() {
    storage.set(Key.COPILOT_NOTIFICATION_BANNER_DISMISSED, "true");
  }

  function handleOpenSettings(event: MouseEvent) {
    // Only persist — setting state here would unmount the banner, and this
    // link with it, before Next gets to navigate. A new-tab click keeps it.
    if (isPlainLeftClick(event)) persistDismissed();
  }

  function handleDismiss() {
    persistDismissed();
    setDismissed(true);
  }

  return (
    <Alert variant="warning" icon={BellRing} aria-live="polite">
      <div className="flex flex-wrap items-center gap-3">
        <AlertDescription className="min-w-[12rem] flex-1">
          Notifications are off. Turn them on in Settings to know when your
          experts finish working, even when you switch tabs.
        </AlertDescription>
        <Button
          as="NextLink"
          variant="primary"
          size="small"
          href="/settings/account"
          onClick={handleOpenSettings}
        >
          Open settings
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={handleDismiss}
          aria-label="Dismiss"
          title="Dismiss"
          className="hover:border-[#FFE4BF] hover:bg-[#FFE4BF]"
        >
          <Icon icon={Cancel01Icon} className="h-4 w-4" />
        </Button>
      </div>
    </Alert>
  );
}
