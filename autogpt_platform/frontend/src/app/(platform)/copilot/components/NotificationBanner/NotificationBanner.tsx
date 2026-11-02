"use client";

import { Button } from "@/components/atoms/Button/Button";
import { createIconComponent, Icon } from "@/components/atoms/Icon/Icon";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { BellRingIcon, Cancel01Icon } from "@hugeicons/core-free-icons";
import { useNotificationBanner } from "./useNotificationBanner";

// Alert's `icon` prop takes a component, not icon data.
const BellRing = createIconComponent(BellRingIcon);

export function NotificationBanner() {
  const { isVisible, handleTurnOn, handleDismiss } = useNotificationBanner();

  if (!isVisible) return null;

  return (
    <Alert variant="warning" icon={BellRing} aria-live="polite">
      <div className="flex flex-wrap items-center gap-3">
        <AlertDescription className="min-w-[12rem] flex-1">
          Get a notification when Otto or your experts finish, even when you
          switch tabs.
        </AlertDescription>
        <Button variant="primary" size="small" onClick={handleTurnOn}>
          Turn on notifications
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={handleDismiss}
          aria-label="Dismiss"
          title="Dismiss"
          // oxlint-disable-next-line design-system/no-arbitrary-values -- peach hover on the banner; the nearest role (warning-border) is a yellow hue, deferred
          className="hover:border-[#FFE4BF] hover:bg-[#FFE4BF]"
        >
          <Icon icon={Cancel01Icon} className="h-4 w-4" />
        </Button>
      </div>
    </Alert>
  );
}
