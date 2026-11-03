import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

const tones = {
  orange: {
    card: "bg-orange/10",
    title: "text-warning-foreground",
    value: "text-warning-foreground",
    detail: "text-warning-foreground",
    hint: "text-orange-foreground",
  },
  error: {
    card: "bg-error-surface",
    title: "text-error-foreground",
    value: "text-error-foreground",
    detail: "text-error-foreground",
    hint: "text-error-foreground",
  },
  yellow: {
    card: "bg-warning-surface",
    title: "text-yellow-foreground",
    value: "text-yellow-foreground",
    detail: "text-yellow-foreground",
    hint: "text-yellow-foreground",
  },
  brand: {
    card: "bg-accent-surface",
    title: "text-accent-strong",
    value: "text-accent-strong",
    detail: "text-accent-strong",
    hint: "text-brand-hover",
  },
  pink: {
    card: "bg-pink/10",
    title: "text-pink-foreground",
    value: "text-pink-foreground",
    detail: "text-pink-foreground",
    hint: "text-pink-foreground",
  },
} as const;

interface Props {
  tone: keyof typeof tones;
  title: string;
  value: ReactNode;
  detail: ReactNode;
  hint: string;
  onClick: () => void;
}

export function DiagnosticAlertCard({
  tone,
  title,
  value,
  detail,
  hint,
  onClick,
}: Props) {
  const colors = tones[tone];
  return (
    <div
      className="cursor-pointer transition-all hover:scale-105"
      onClick={onClick}
    >
      <div className={cn("rounded-large p-6 shadow-md", colors.card)}>
        <div className="flex flex-col space-y-1.5 p-6 pb-3">
          <h3
            className={cn(
              "leading-none font-semibold tracking-tight",
              colors.title,
            )}
          >
            {title}
          </h3>
        </div>
        <div className="p-6 pt-0">
          <p className={cn("text-heading font-bold", colors.value)}>{value}</p>
          <p className={cn("text-body", colors.detail)}>{detail}</p>
          <p className={cn("mt-2 text-caption", colors.hint)}>{hint}</p>
        </div>
      </div>
    </div>
  );
}
