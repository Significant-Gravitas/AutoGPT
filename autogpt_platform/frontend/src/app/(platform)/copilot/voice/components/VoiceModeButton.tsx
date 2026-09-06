"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { StopIcon, VoiceIcon } from "@hugeicons/core-free-icons";

interface Props {
  isActive: boolean;
  /** AutoPilot is talking: the same click cuts it off, so say so. */
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
      variant="icon"
      size="icon"
      aria-label={
        speaking ? "Stop" : isActive ? "Leave voice mode" : "Talk to AutoPilot"
      }
      aria-pressed={isActive}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "border-transparent bg-transparent text-black shadow-none hover:border-transparent hover:bg-zinc-100 hover:text-black",
        className,
        disabled && "opacity-40",
        isActive && "bg-zinc-900 text-white hover:bg-zinc-800 hover:text-white",
      )}
    >
      <Icon icon={speaking ? StopIcon : VoiceIcon} className="h-4 w-4" />
    </Button>
  );
}
