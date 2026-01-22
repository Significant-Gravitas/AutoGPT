import { cn } from "@/lib/utils";
import { ReactNode } from "react";

export interface AIChatBubbleProps {
  children: ReactNode;
  className?: string;
}

export function AIChatBubble({ children, className }: AIChatBubbleProps) {
  return (
    <div className={cn("text-left text-[1rem] leading-relaxed", className)}>
      {children}
    </div>
  );
}
