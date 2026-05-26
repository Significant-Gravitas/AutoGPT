"use client";

import { Switch } from "@/components/atoms/Switch/Switch";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { toast } from "@/components/molecules/Toast/use-toast";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Bell, BellRinging, BellSlash } from "@phosphor-icons/react";
import { useCopilotUIStore } from "../../../../store";

export function NotificationToggle() {
  const {
    isNotificationsEnabled,
    setNotificationsEnabled,
    isSoundEnabled,
    toggleSound,
  } = useCopilotUIStore();

  async function handleToggleNotifications() {
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
    const permission = await Notification.requestPermission();
    if (permission === "granted") {
      setNotificationsEnabled(true);
    } else {
      toast({
        title: "Notifications blocked",
        description:
          "Please allow notifications in your browser settings to enable this feature.",
        variant: "destructive",
      });
    }
  }

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Notification settings">
          {!isNotificationsEnabled ? (
            <BellSlash className="!size-5" />
          ) : isSoundEnabled ? (
            <BellRinging className="!size-5" />
          ) : (
            <Bell className="!size-5" />
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-56 p-3">
        <div className="flex flex-col gap-3">
          <label className="flex items-center justify-between">
            <span className="text-sm text-zinc-700">Notifications</span>
            <Switch
              checked={isNotificationsEnabled}
              onCheckedChange={handleToggleNotifications}
            />
          </label>
          <label className="flex items-center justify-between">
            <span
              className={cn(
                "text-sm text-zinc-700",
                !isNotificationsEnabled && "opacity-50",
              )}
            >
              Sound
            </span>
            <Switch
              checked={isSoundEnabled && isNotificationsEnabled}
              onCheckedChange={toggleSound}
              disabled={!isNotificationsEnabled}
            />
          </label>
        </div>
      </PopoverContent>
    </Popover>
  );
}
