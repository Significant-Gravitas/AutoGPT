"use client";

import { cn } from "@/lib/utils";
import { Flask } from "@phosphor-icons/react";

interface Props {
  isDryRun: boolean;
  isStreaming: boolean;
  hasSession: boolean;
  onToggle: () => void;
}

export function DryRunToggleButton({
  isDryRun,
  isStreaming,
  hasSession,
  onToggle,
}: Props) {
  const isDisabled = isStreaming || hasSession;
  return (
    <button
      type="button"
      aria-pressed={isDryRun}
      disabled={isDisabled}
      onClick={onToggle}
      className={cn(
        "inline-flex min-h-11 min-w-11 items-center justify-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors",
        isDryRun
          ? "bg-amber-100 text-amber-900 hover:bg-amber-200"
          : "text-neutral-500 hover:bg-neutral-100 hover:text-neutral-700",
        isDisabled && "cursor-not-allowed opacity-50",
      )}
      aria-label={isDryRun ? "Disable Test mode" : "Enable Test mode"}
      title={
        hasSession
          ? "Start a new chat to change modes"
          : isStreaming
            ? "Cannot change mode while streaming"
            : isDryRun
              ? "Test mode ON — new sessions use dry_run=true (click to disable)"
              : "Enable Test mode — new sessions will use dry_run=true"
      }
    >
      <Flask size={14} />
      {isDryRun && "Test"}
    </button>
  );
}
