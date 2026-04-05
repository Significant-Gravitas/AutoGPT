"use client";

import { cn } from "@/lib/utils";
import { Brain, Lightning } from "@phosphor-icons/react";

type CopilotMode = "extended_thinking" | "fast";

interface Props {
  mode: CopilotMode;
  isStreaming: boolean;
  onToggle: () => void;
}

export function ModeToggleButton({ mode, isStreaming, onToggle }: Props) {
  const isExtended = mode === "extended_thinking";
  return (
    <button
      type="button"
      aria-pressed={isExtended}
      disabled={isStreaming}
      onClick={onToggle}
      className={cn(
        "inline-flex min-h-11 min-w-11 items-center justify-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors",
        isExtended
          ? "bg-purple-100 text-purple-700 hover:bg-purple-200 dark:bg-purple-900/30 dark:text-purple-300"
          : "bg-amber-100 text-amber-700 hover:bg-amber-200 dark:bg-amber-900/30 dark:text-amber-300",
        isStreaming && "cursor-not-allowed opacity-50",
      )}
      aria-label={
        isExtended ? "Switch to Fast mode" : "Switch to Extended Thinking mode"
      }
      title={
        isStreaming
          ? "Mode cannot be changed while streaming"
          : isExtended
            ? "Extended Thinking mode — deeper reasoning (click to switch to Fast mode)"
            : "Fast mode — quicker responses (click to switch to Extended Thinking)"
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
  );
}
