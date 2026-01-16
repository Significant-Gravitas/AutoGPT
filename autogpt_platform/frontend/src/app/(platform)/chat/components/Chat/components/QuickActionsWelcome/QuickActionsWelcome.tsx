"use client";

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
      className={cn("flex flex-1 items-center justify-center p-8", className)}
    >
      <div className="w-full max-w-3xl">
        <div className="mb-12 text-center">
          <Text
            variant="h2"
            className="mb-3 text-2xl font-semibold text-zinc-900"
          >
            {title}
          </Text>
          <Text variant="body" className="text-zinc-500">
            {description}
          </Text>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {actions.map((action) => {
            // Use slate theme for all cards
            const theme = {
              bg: "bg-slate-50/10",
              border: "border-slate-100",
              hoverBg: "hover:bg-slate-50/20",
              hoverBorder: "hover:border-slate-200",
              gradient: "from-slate-200/20 via-slate-300/10 to-transparent",
              text: "text-slate-900",
              hoverText: "group-hover:text-slate-900",
            };

            return (
              <button
                key={action}
                onClick={() => onActionClick(action)}
                disabled={disabled}
                className={cn(
                  "group relative overflow-hidden rounded-xl border p-5 text-left backdrop-blur-xl",
                  "transition-all duration-200",
                  theme.bg,
                  theme.border,
                  theme.hoverBg,
                  theme.hoverBorder,
                  "hover:shadow-sm",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:ring-offset-2",
                  "disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:shadow-none",
                )}
              >
                {/* Gradient flare background */}
                <div
                  className={cn(
                    "absolute inset-0 bg-gradient-to-br",
                    theme.gradient,
                  )}
                />

                <Text
                  variant="body"
                  className={cn(
                    "relative z-10 font-medium",
                    theme.text,
                    theme.hoverText,
                  )}
                >
                  {action}
                </Text>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
