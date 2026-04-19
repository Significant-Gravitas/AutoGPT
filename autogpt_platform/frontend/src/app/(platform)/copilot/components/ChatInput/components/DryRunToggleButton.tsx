"use client";

import { cn } from "@/lib/utils";
import { Flask } from "@phosphor-icons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// This button is only rendered on NEW chats (no active session).
// Once a session exists, it is hidden — the session's dry_run flag is
// immutable and reflected in the banner in CopilotPage.tsx instead.
// Do NOT add readOnly/hasSession handling here; hide it at the call site.
interface Props {
  isDryRun: boolean;
  onToggle: () => void;
}

export function DryRunToggleButton({ isDryRun, onToggle }: Props) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-pressed={isDryRun}
          onClick={onToggle}
          className={cn(
            "inline-flex h-9 items-center justify-center gap-1 rounded-full border border-neutral-200 bg-white px-2.5 text-xs font-medium shadow-sm transition-colors hover:bg-neutral-50",
            isDryRun
              ? "text-amber-900"
              : "text-neutral-500 hover:text-neutral-700",
          )}
          aria-label={isDryRun ? "Test mode active" : "Enable Test mode"}
        >
          <Flask size={14} />
          <span className="hidden sm:inline">
            {isDryRun ? "Test mode enabled" : "Enable test mode"}
          </span>
        </button>
      </TooltipTrigger>
      <TooltipContent>
        {isDryRun
          ? "Test mode on — new sessions run without performing real actions (click to turn off)."
          : "Turn on test mode to try prompts without performing real actions."}
      </TooltipContent>
    </Tooltip>
  );
}
