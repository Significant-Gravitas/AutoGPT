import React from "react";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

export interface QuickActionsWelcomeProps {
  title: string;
  description: string;
  actions: string[];
  onActionClick: (action: string) => void;
  disabled?: boolean;
  className?: string;
}

export function QuickActionsWelcome({
  title,
  description,
  actions,
  onActionClick,
  disabled = false,
  className,
}: QuickActionsWelcomeProps) {
  return (
    <div
      className={cn("flex flex-1 items-center justify-center p-4", className)}
    >
      <div className="max-w-2xl text-center">
        <Text
          variant="h2"
          className="mb-4 text-3xl font-bold text-zinc-900 dark:text-zinc-100"
        >
          {title}
        </Text>
        <Text variant="body" className="mb-8 text-zinc-600 dark:text-zinc-400">
          {description}
        </Text>
        <div className="grid gap-2 sm:grid-cols-2">
          {actions.map((action) => (
            <button
              key={action}
              onClick={() => onActionClick(action)}
              disabled={disabled}
              className="rounded-lg border border-zinc-200 bg-white p-4 text-left text-sm hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800"
            >
              {action}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
