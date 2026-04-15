"use client";

import { cn } from "@/lib/utils";
import { Flask } from "@phosphor-icons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface Props {
  isDryRun: boolean;
  isStreaming: boolean;
  readOnly?: boolean;
  onToggle: () => void;
}

export function DryRunToggleButton({
  isDryRun,
  isStreaming,
  readOnly = false,
  onToggle,
}: Props) {
  const isDisabled = isStreaming || readOnly;

  function handleClick() {
    if (isDisabled) return;
    onToggle();
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-pressed={isDryRun}
          onClick={handleClick}
          className={cn(
            "inline-flex h-9 items-center justify-center gap-1 rounded-full border border-neutral-200 bg-white px-2.5 text-xs font-medium shadow-sm transition-colors hover:bg-neutral-50",
            isDryRun
              ? "text-amber-900"
              : "text-neutral-500 hover:text-neutral-700",
            isDisabled && "cursor-default opacity-70",
          )}
          aria-label={isDryRun ? "Test mode active" : "Enable Test mode"}
        >
          <Flask size={14} />
          <span className="hidden sm:inline">
            {isDryRun ? "Test mode enabled" : "Enable test mode"}
          </span>
        </button>
      </TooltipTrigger>
      {isDryRun && (
        <TooltipContent>
          Test mode — new sessions use dry_run=true
        </TooltipContent>
      )}
    </Tooltip>
  );
}
