"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { BrainIcon, LightningIcon, LockIcon } from "@phosphor-icons/react";
import type { CopilotMode } from "../../../store";
import { ToggleChip } from "@/components/atoms/ToggleChip/ToggleChip";

interface Props {
  mode: CopilotMode;
  onToggle: () => void;
  /** When true the mode is locked to Extended Thinking (agent build flow). */
  pinned?: boolean;
  /** "pill" is the pre-brain-dump footer style; "chip" lives in the tray. */
  variant?: "chip" | "pill";
}

export function ModeToggleButton({
  mode,
  onToggle,
  pinned = false,
  variant = "chip",
}: Props) {
  const isExtended = mode === "extended_thinking";
  const tooltipText = pinned
    ? "Extended Thinking is required while building an agent"
    : isExtended
      ? "Extended Thinking — deeper reasoning (click to switch to Fast)"
      : "Fast mode — quicker responses (click to switch to Thinking)";

  const ariaLabel = pinned
    ? "Mode locked to Extended Thinking while building an agent"
    : isExtended
      ? "Switch to Fast mode"
      : "Switch to Extended Thinking mode";

  function getIcon() {
    if (pinned) return <LockIcon size={14} />;
    return isExtended ? <BrainIcon size={14} /> : <LightningIcon size={14} />;
  }

  if (variant === "pill") {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-pressed={isExtended}
            aria-disabled={pinned}
            onClick={onToggle}
            className={cn(
              "ml-2 inline-flex h-9 min-w-[6rem] items-center justify-center gap-1 rounded-full border border-neutral-200 bg-white px-2.5 text-xs font-medium shadow-sm transition-colors hover:bg-neutral-50",
              isExtended ? "text-purple-500" : "text-orange-600",
              pinned && "cursor-not-allowed opacity-70 hover:bg-white",
            )}
            aria-label={ariaLabel}
          >
            {getIcon()}
            {pinned || isExtended ? "Thinking" : "Fast"}
          </button>
        </TooltipTrigger>
        <TooltipContent>{tooltipText}</TooltipContent>
      </Tooltip>
    );
  }

  return (
    <ToggleChip
      icon={getIcon()}
      label={pinned || isExtended ? "Thinking" : "Fast"}
      tooltip={tooltipText}
      ariaLabel={ariaLabel}
      pressed={isExtended}
      onToggle={onToggle}
      locked={pinned}
      className="sm:min-w-[5.5rem]"
    />
  );
}
