"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Alert01Icon } from "@hugeicons/core-free-icons";

import { useNotificationSettings } from "./useNotificationSettings";

interface RowProps {
  label: string;
  description?: string;
  checked: boolean;
  disabled?: boolean;
  onCheckedChange: () => void;
}

function SettingRow({
  label,
  description,
  checked,
  disabled,
  onCheckedChange,
}: RowProps) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div className={cn("flex flex-col", disabled && "opacity-50")}>
        <Text variant="body-medium" as="span" className="text-textBlack">
          {label}
        </Text>
        {description ? (
          <Text variant="small" as="span" className="text-zinc-500">
            {description}
          </Text>
        ) : null}
      </div>
      <Switch
        checked={checked}
        disabled={disabled}
        onCheckedChange={onCheckedChange}
        aria-label={label}
      />
    </div>
  );
}

export function NotificationSettingsControls() {
  const {
    isNotificationsEnabled,
    isSoundEnabled,
    isSupported,
    isBlocked,
    toggleNotifications,
    toggleSound,
  } = useNotificationSettings();

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
        <div className="flex items-start gap-2 rounded-lg bg-zinc-100 px-3 py-2">
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
