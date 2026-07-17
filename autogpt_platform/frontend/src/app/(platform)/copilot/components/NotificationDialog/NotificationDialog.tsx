"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Key, storage } from "@/services/storage/local-storage";
import { BellRinging } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { useCopilotUIStore } from "../../store";

export function NotificationDialog() {
  const {
    showNotificationDialog,
    setShowNotificationDialog,
    setNotificationsEnabled,
    isNotificationsEnabled,
  } = useCopilotUIStore();

  const [dismissed, setDismissed] = useState(
    () => storage.get(Key.COPILOT_NOTIFICATION_DIALOG_DISMISSED) === "true",
  );
  const [permission, setPermission] = useState(() =>
    typeof Notification !== "undefined" ? Notification.permission : "denied",
  );

  // Re-read dismissed flag when notifications are toggled off (e.g. clearCopilotLocalData)
  useEffect(() => {
    if (!isNotificationsEnabled) {
      setDismissed(
        storage.get(Key.COPILOT_NOTIFICATION_DIALOG_DISMISSED) === "true",
      );
    }
  }, [isNotificationsEnabled]);

  const shouldShowAuto =
    typeof Notification !== "undefined" &&
    permission === "default" &&
    !dismissed;

  const isOpen = showNotificationDialog || shouldShowAuto;

  function handleEnable() {
    if (typeof Notification === "undefined") {
      handleDismiss();
      return;
    }
    Notification.requestPermission().then((result) => {
      setPermission(result);
      if (result === "granted") {
        setNotificationsEnabled(true);
        handleDismiss();
      }
    });
  }

  function handleDismiss() {
    storage.set(Key.COPILOT_NOTIFICATION_DIALOG_DISMISSED, "true");
    setDismissed(true);
    setShowNotificationDialog(false);
  }

  return (
    <Dialog
      title="Stay in the loop"
      styling={{ maxWidth: "28rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) handleDismiss();
        },
      }}
      onClose={handleDismiss}
    >
      <Dialog.Content>
        <div className="flex flex-col items-center gap-4 py-2">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-violet-100">
            <BellRinging className="h-6 w-6 text-violet-600" weight="fill" />
          </div>
          <Text variant="body" className="text-center text-neutral-600">
            AutoPilot can notify you when a response is ready, even if you
            switch tabs or close this page. Enable notifications so you never
            miss one.
          </Text>
        </div>
        <Dialog.Footer className="justify-center">
          <Button variant="secondary" onClick={handleDismiss}>
            Not now
          </Button>
          <Button variant="primary" onClick={handleEnable}>
            Enable notifications
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
