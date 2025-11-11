import React from "react";
import { WrenchIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { getToolActionPhrase } from "@/app/(platform)/chat/helpers";

export interface ToolCallMessageProps {
  toolName: string;
  className?: string;
}

export function ToolCallMessage({ toolName, className }: ToolCallMessageProps) {
  return (
    <div
      className={cn(
        "mx-10 max-w-[70%] overflow-hidden rounded-lg border transition-all duration-200",
        "border-neutral-200 dark:border-neutral-700",
        "bg-white dark:bg-neutral-900",
        "animate-in fade-in-50 slide-in-from-top-1",
        className,
      )}
    >
      {/* Header */}
      <div
        className={cn(
          "flex items-center justify-between px-3 py-2",
          "bg-gradient-to-r from-neutral-50 to-neutral-100 dark:from-neutral-800/20 dark:to-neutral-700/20",
        )}
      >
        <div className="flex items-center gap-2 overflow-hidden">
          <WrenchIcon
            size={16}
            weight="bold"
            className="flex-shrink-0 text-neutral-500 dark:text-neutral-400"
          />
          <span className="relative inline-block overflow-hidden text-sm font-medium text-neutral-700 dark:text-neutral-300">
            {getToolActionPhrase(toolName)}...
            <span
              className={cn(
                "absolute inset-0 bg-gradient-to-r from-transparent via-white/50 to-transparent",
                "dark:via-white/20",
                "animate-shimmer",
              )}
            />
          </span>
        </div>
      </div>
    </div>
  );
}
