"use client";

import { cn } from "@/lib/utils";

interface Props {
  isSourceView: boolean;
  onToggle: (isSource: boolean) => void;
}

export function SourceToggle({ isSourceView, onToggle }: Props) {
  return (
    <div className="flex items-center rounded-md border border-zinc-200 bg-zinc-50 p-0.5 text-xs font-medium">
      <button
        type="button"
        aria-pressed={!isSourceView}
        className={cn(
          "rounded px-2 py-1 transition-colors",
          !isSourceView
            ? "bg-white text-zinc-900 shadow-sm"
            : "text-zinc-500 hover:text-zinc-700",
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
            ? "bg-white text-zinc-900 shadow-sm"
            : "text-zinc-500 hover:text-zinc-700",
        )}
        onClick={() => onToggle(true)}
      >
        Source
      </button>
    </div>
  );
}
