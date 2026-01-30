import { cn } from "@/lib/utils";
import { ReactNode } from "react";

export interface UserChatBubbleProps {
  children: ReactNode;
  className?: string;
}

export function UserChatBubble({ children, className }: UserChatBubbleProps) {
  return (
    <div
      className={cn(
        "group relative min-w-20 overflow-hidden rounded-xl bg-purple-100 px-3 text-left text-[1rem] leading-relaxed transition-all duration-500 ease-in-out",
        className,
      )}
      style={{
        borderBottomRightRadius: 0,
      }}
    >
      <div className="relative z-10 text-slate-900 transition-all duration-500 ease-in-out">
        {children}
      </div>
    </div>
  );
}
