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
        "min-w-20 rounded-[20px] px-6 py-2.5 text-sm leading-relaxed",
        variant === "user" && "bg-zinc-700 text-right text-neutral-50",
        variant === "assistant" && "bg-zinc-100 text-left text-neutral-900",
        className,
      )}
    >
      {children}
    </div>
  );
}
