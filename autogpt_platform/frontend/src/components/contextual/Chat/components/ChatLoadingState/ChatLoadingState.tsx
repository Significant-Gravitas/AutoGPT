import React from "react";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowClockwiseIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";

export interface ChatLoadingStateProps {
  message?: string;
  className?: string;
}

export function ChatLoadingState({
  message = "Loading...",
  className,
}: ChatLoadingStateProps) {
  return (
    <div
      className={cn("flex flex-1 items-center justify-center p-6", className)}
    >
      <div className="flex flex-col items-center gap-4 text-center">
        <ArrowClockwiseIcon
          size={32}
          weight="bold"
          className="animate-spin text-purple-500"
        />
        <Text variant="body" className="text-zinc-600 dark:text-zinc-400">
          {message}
        </Text>
      </div>
    </div>
  );
}
