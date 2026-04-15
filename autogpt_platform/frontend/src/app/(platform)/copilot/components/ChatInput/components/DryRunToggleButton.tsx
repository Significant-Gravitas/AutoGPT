"use client";

import { cn } from "@/lib/utils";
import { Flask } from "@phosphor-icons/react";

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
    <button
      type="button"
      aria-pressed={isDryRun}
      onClick={onToggle}
      className={cn(
        "inline-flex min-h-11 min-w-11 items-center justify-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors",
        isDryRun
          ? "bg-amber-100 text-amber-900 hover:bg-amber-200"
          : "text-neutral-500 hover:bg-neutral-100 hover:text-neutral-700",
      )}
      aria-label={
        isDryRun ? "Test mode active — click to disable" : "Enable Test mode"
      }
      title={
        isDryRun
          ? "Test mode ON — new chats run agents as simulation (click to disable)"
          : "Enable Test mode — new chats will run agents as simulation"
      }
    >
      <Flask size={14} />
      {isDryRun && "Test"}
    </button>
  );
}
