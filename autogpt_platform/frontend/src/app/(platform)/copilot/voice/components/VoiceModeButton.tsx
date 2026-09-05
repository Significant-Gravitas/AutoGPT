"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { VoiceIcon } from "@hugeicons/core-free-icons";

interface Props {
  isActive: boolean;
  disabled?: boolean;
  onClick: () => void;
  className?: string;
}

export function VoiceModeButton({
  isActive,
  disabled = false,
  onClick,
  className,
}: Props) {
  return (
    <Button
      type="button"
      variant="icon"
      size="icon"
      aria-label={isActive ? "Leave voice mode" : "Talk to AutoPilot"}
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
      <Icon icon={VoiceIcon} className="h-4 w-4" />
    </Button>
  );
}
