"use client";

import { cn } from "@/lib/utils";
import { Flask } from "@phosphor-icons/react";

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
  return (
    <button
      type="button"
      aria-pressed={isDryRun}
      disabled={isDisabled}
      onClick={readOnly ? undefined : onToggle}
      className={cn(
        "inline-flex min-h-11 min-w-11 items-center justify-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors",
        isDryRun
          ? "bg-amber-100 text-amber-900 hover:bg-amber-200"
          : "text-neutral-500 hover:bg-neutral-100 hover:text-neutral-700",
        isDisabled && "cursor-default opacity-70",
      )}
      aria-label={isDryRun ? "Test mode active" : "Enable Test mode"}
      title={
        readOnly
          ? "Test mode active for this session"
          : isStreaming
            ? "Cannot change mode while streaming"
            : isDryRun
              ? "Test mode ON — click to disable"
              : "Enable Test mode — agents will run as dry-run"
      }
    >
      <Flask size={14} />
      {isDryRun && "Test"}
    </button>
  );
}
