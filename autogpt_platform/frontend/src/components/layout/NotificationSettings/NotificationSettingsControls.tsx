"use client";

import { useId } from "react";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Alert01Icon } from "@hugeicons/core-free-icons";

import { SettingRow } from "./components/SettingRow/SettingRow";
import { useNotificationSettings } from "./useNotificationSettings";

export function NotificationSettingsControls() {
  const {
    isNotificationsEnabled,
    isSoundEnabled,
    isSupported,
    isBlocked,
    toggleNotifications,
    toggleSound,
  } = useNotificationSettings();
  const blockedExplainerId = useId();

  if (!isSupported) {
    return (
      <Text variant="small" as="p" className="text-zinc-500">
        This browser doesn&apos;t support notifications.
      </Text>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <SettingRow
        label="Notifications"
        description="Get told when your experts finish, even in another tab."
        checked={isNotificationsEnabled}
        disabled={isBlocked}
        describedBy={isBlocked ? blockedExplainerId : undefined}
        onCheckedChange={toggleNotifications}
      />
      <SettingRow
        label="Sound"
        description="Play a chime with each one."
        checked={isSoundEnabled && isNotificationsEnabled}
        disabled={!isNotificationsEnabled}
        onCheckedChange={toggleSound}
      />

      {/* A browser-level denial is the one state we can't fix from in here —
          say so rather than leaving a switch that silently refuses to move. */}
      {isBlocked ? (
        <div
          id={blockedExplainerId}
          className="flex items-start gap-2 rounded-lg bg-zinc-100 px-3 py-2"
        >
          <Icon
            icon={Alert01Icon}
            className="mt-0.5 size-4 shrink-0 text-zinc-500"
          />
          <Text variant="small" as="p" className="text-zinc-600">
            Your browser is blocking notifications for AutoGPT. Allow them in
            its site settings to switch this back on.
          </Text>
        </div>
      ) : null}
    </div>
  );
}
