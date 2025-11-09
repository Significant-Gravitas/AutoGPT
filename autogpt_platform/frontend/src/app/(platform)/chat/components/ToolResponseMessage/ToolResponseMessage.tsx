import React from "react";
import { WrenchIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { getToolActionPhrase } from "@/app/(platform)/chat/helpers";

export interface ToolResponseMessageProps {
  toolName: string;
  success?: boolean;
  className?: string;
}

export function ToolResponseMessage({
  toolName,
  success = true,
  className,
}: ToolResponseMessageProps) {
  return (
    <div
      className={cn(
        "mx-10 max-w-[70%] overflow-hidden rounded-lg border transition-all duration-200",
        success
          ? "border-neutral-200 dark:border-neutral-700"
          : "border-red-200 dark:border-red-800",
        "bg-white dark:bg-neutral-900",
        "animate-in fade-in-50 slide-in-from-top-1",
        className,
      )}
    >
      {/* Header */}
      <div
        className={cn(
          "flex items-center justify-between px-3 py-2",
          "bg-gradient-to-r",
          success
            ? "from-neutral-50 to-neutral-100 dark:from-neutral-800/20 dark:to-neutral-700/20"
            : "from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20",
        )}
      >
        <div className="flex items-center gap-2">
          <WrenchIcon
            size={16}
            weight="bold"
            className="text-neutral-500 dark:text-neutral-400"
          />
          <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
            {getToolActionPhrase(toolName)}...
          </span>
        </div>
      </div>
    </div>
  );
}
