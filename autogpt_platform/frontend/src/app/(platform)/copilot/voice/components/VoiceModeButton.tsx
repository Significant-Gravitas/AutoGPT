"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { StopIcon, VoiceIcon } from "@hugeicons/core-free-icons";

interface Props {
  isActive: boolean;
  /** Otto is talking: the same click cuts it off, so say so. */
  speaking?: boolean;
  disabled?: boolean;
  onClick: () => void;
  className?: string;
}

export function VoiceModeButton({
  isActive,
  speaking = false,
  disabled = false,
  onClick,
  className,
}: Props) {
  return (
    <Button
      type="button"
      variant="tray"
      size="icon"
      aria-label={
        speaking ? "Stop" : isActive ? "Leave voice mode" : "Start voice mode"
      }
      aria-pressed={isActive}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        className,
        isActive &&
          "bg-action-hover text-action-foreground hover:bg-action hover:text-action-foreground",
      )}
    >
      <Icon icon={speaking ? StopIcon : VoiceIcon} className="h-4 w-4" />
    </Button>
  );
}
