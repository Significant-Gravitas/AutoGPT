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
  const userTheme = {
    bg: "bg-purple-100",
    border: "border-purple-100",
    text: "text-slate-900",
  };

  const assistantTheme = {
    bg: "bg-slate-50/20",
    border: "border-slate-100",
    gradient: "from-slate-200/20 via-slate-300/10 to-transparent",
    text: "text-slate-900",
  };

  const theme = variant === "user" ? userTheme : assistantTheme;

  return (
    <div
      className={cn(
        "group relative min-w-20 overflow-hidden rounded-xl border px-6 py-2.5 text-sm leading-relaxed backdrop-blur-xl transition-all duration-500 ease-in-out",
        theme.bg,
        theme.border,
        variant === "user" && "text-right",
        variant === "assistant" && "text-left",
        className,
      )}
    >
      {/* Gradient flare background */}
      <div className={cn("absolute inset-0 bg-gradient-to-br")} />
      <div
        className={cn(
          "relative z-10 transition-all duration-500 ease-in-out",
          theme.text,
        )}
      >
        {children}
      </div>
    </div>
  );
}
