import { cn } from "@/lib/utils";
import { ReactNode } from "react";

export interface MessageBubbleProps {
  children: ReactNode;
  variant: "user" | "assistant";
  className?: string;
}

export function MessageBubble({
  children,
  variant,
  className,
}: MessageBubbleProps) {
  return (
    <div
      className={cn(
        "rounded-lg px-4 py-3 text-sm",
        variant === "user" && "bg-violet-600 text-white dark:bg-violet-500",
        variant === "assistant" &&
          "border border-neutral-200 bg-white dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-100",
        className,
      )}
    >
      {children}
    </div>
  );
}
