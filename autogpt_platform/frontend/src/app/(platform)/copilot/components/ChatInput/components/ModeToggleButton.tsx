"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ToggleChip } from "@/components/atoms/ToggleChip/ToggleChip";
import { cn } from "@/lib/utils";
import { BrainIcon, FlashIcon, LockIcon } from "@hugeicons/core-free-icons";
import type { CopilotMode } from "../../../store";

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
    if (pinned) return <Icon icon={LockIcon} size={14} />;
    return isExtended ? (
      <Icon icon={BrainIcon} size={14} />
    ) : (
      <Icon icon={FlashIcon} size={14} />
    );
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
              "inline-flex h-8 items-center justify-center gap-1.5 rounded-full px-2.5 text-[13px] font-medium text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700",
              pinned && "cursor-not-allowed opacity-70 hover:bg-transparent",
            )}
            aria-label={ariaLabel}
          >
            {getIcon()}
            <span className="hidden sm:inline">
              {pinned || isExtended ? "Thinking" : "Fast"}
            </span>
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
