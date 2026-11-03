"use client";

import { cn } from "@/lib/utils";

interface Props {
  isSourceView: boolean;
  onToggle: (isSource: boolean) => void;
}

export function SourceToggle({ isSourceView, onToggle }: Props) {
  return (
    <div className="flex items-center rounded-md border border-border-base bg-surface-muted p-0.5 text-xs font-medium">
      <button
        type="button"
        aria-pressed={!isSourceView}
        className={cn(
          "rounded px-2 py-1 transition-colors",
          !isSourceView
            ? "bg-surface text-foreground-strong shadow-sm"
            : "text-foreground-tertiary hover:text-foreground-body",
        )}
        onClick={() => onToggle(false)}
      >
        Preview
      </button>
      <button
        type="button"
        aria-pressed={isSourceView}
        className={cn(
          "rounded px-2 py-1 transition-colors",
          isSourceView
            ? "bg-surface text-foreground-strong shadow-sm"
            : "text-foreground-tertiary hover:text-foreground-body",
        )}
        onClick={() => onToggle(true)}
      >
        Source
      </button>
    </div>
  );
}
