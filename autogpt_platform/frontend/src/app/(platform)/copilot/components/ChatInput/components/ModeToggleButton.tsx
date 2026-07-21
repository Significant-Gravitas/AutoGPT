"use client";

import { cn } from "@/lib/utils";
import { Brain, Lightning, Lock } from "@phosphor-icons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { CopilotMode } from "../../../store";

interface Props {
  mode: CopilotMode;
  onToggle: () => void;
  pinned?: boolean;
}

export function ModeToggleButton({ mode, onToggle, pinned = false }: Props) {
  const isExtended = mode === "extended_thinking";

  const tooltipText = pinned
    ? "Locked to Extended Thinking — building sessions stay on this engine"
    : isExtended
      ? "Extended Thinking — deeper reasoning (click to switch to Fast)"
      : "Fast mode — quicker responses (click to switch to Thinking)";

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
          aria-label={
            pinned
              ? "Mode locked to Extended Thinking while building an agent"
              : isExtended
                ? "Switch to Fast mode"
                : "Switch to Extended Thinking mode"
          }
        >
          {pinned ? (
            <>
              <Lock size={14} />
              Thinking
            </>
          ) : isExtended ? (
            <>
              <Brain size={14} />
              Thinking
            </>
          ) : (
            <>
              <Lightning size={14} />
              Fast
            </>
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent>{tooltipText}</TooltipContent>
    </Tooltip>
  );
}
