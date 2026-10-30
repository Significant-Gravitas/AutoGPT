import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

/* oxlint-disable design-system/no-raw-colors -- orange, yellow and pink have no role; amber or red would change the hue (deferred) */
const tones = {
  orange: {
    card: "bg-orange-50",
    title: "text-orange-800",
    value: "text-orange-900",
    detail: "text-orange-700",
    hint: "text-orange-600",
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
    title: "text-yellow-800",
    value: "text-yellow-900",
    detail: "text-yellow-700",
    hint: "text-yellow-600",
  },
  brand: {
    card: "bg-accent-surface",
    title: "text-accent-strong",
    value: "text-accent-strong",
    detail: "text-accent-strong",
    hint: "text-brand-hover",
  },
  pink: {
    card: "bg-pink-50",
    title: "text-pink-800",
    value: "text-pink-900",
    detail: "text-pink-700",
    hint: "text-pink-600",
  },
} as const;
/* oxlint-enable design-system/no-raw-colors */

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
          <p className={cn("text-3xl font-bold", colors.value)}>{value}</p>
          <p className={cn("text-sm", colors.detail)}>{detail}</p>
          <p className={cn("mt-2 text-xs", colors.hint)}>{hint}</p>
        </div>
      </div>
    </div>
  );
}
