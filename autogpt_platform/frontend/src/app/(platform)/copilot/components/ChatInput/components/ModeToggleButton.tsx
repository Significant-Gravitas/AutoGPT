"use client";

import { cn } from "@/lib/utils";
import { Brain, Lightning } from "@phosphor-icons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { CopilotMode } from "../../../store";

interface Props {
  mode: CopilotMode;
  onToggle: () => void;
}

export function ModeToggleButton({ mode, onToggle }: Props) {
  const isExtended = mode === "extended_thinking";

  const tooltipText = isExtended
    ? "Extended Thinking — deeper reasoning (click to switch to Fast)"
    : "Fast mode — quicker responses (click to switch to Thinking)";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-pressed={isExtended}
          onClick={onToggle}
          className={cn(
            "ml-2 inline-flex h-9 items-center justify-center gap-1 rounded-full border border-neutral-200 bg-white px-2.5 text-xs font-medium shadow-sm transition-colors hover:bg-neutral-50",
            isExtended ? "text-purple-900" : "text-amber-900",
          )}
          aria-label={
            isExtended
              ? "Switch to Fast mode"
              : "Switch to Extended Thinking mode"
          }
        >
          {isExtended ? (
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
