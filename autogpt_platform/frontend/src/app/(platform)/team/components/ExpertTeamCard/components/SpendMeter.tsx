import { Progress } from "@/components/atoms/Progress/Progress";
import { creditsToUsdLabel } from "@/lib/credits";
import { cn } from "@/lib/utils";
import type { CSSProperties } from "react";

interface Props {
  spent: number;
  budget: number;
  muted?: boolean;
  /** Tints the track and fill with darker shades of this colour (e.g. the
   *  expert's cover) instead of grey. */
  color?: string;
}

export function SpendMeter({ spent, budget, muted, color }: Props) {
  const ratio = budget > 0 ? Math.min(Math.max(spent / budget, 0), 1) : 0;
  const isOverBudget = spent > budget;
  const clampedSpent = Math.min(Math.max(spent, 0), Math.max(budget, 0));
  const valueText = `${creditsToUsdLabel(spent)} of ${creditsToUsdLabel(budget)} spent this week${
    isOverBudget ? " (over budget)" : ""
  }`;

  return (
    /* oxlint-disable design-system/no-restyle, design-system/no-inline-styles -- Progress has no tinted or muted variant (src/components is P9's); the tint mixes the expert's data colour */
    <Progress
      value={ratio * 100}
      role="progressbar"
      aria-label="Weekly spend"
      aria-valuenow={clampedSpent}
      aria-valuemin={0}
      aria-valuemax={budget}
      aria-valuetext={valueText}
      style={
        color
          ? ({
              "--meter-track": `color-mix(in srgb, ${color} 45%, rgb(var(--color-surface)))`,
              "--meter-fill": `color-mix(in srgb, ${color} 60%, rgb(var(--color-action)))`,
            } as CSSProperties)
          : undefined
      }
      className={cn(
        "h-1.5 w-full bg-surface-hover",
        color && "h-2 ![background-color:var(--meter-track)]",
        isOverBudget
          ? "[&>div]:bg-error"
          : color
            ? "[&>div]:![background-color:var(--meter-fill)]"
            : "[&>div]:bg-foreground-subtle",
        muted && "opacity-50",
      )}
    />
    /* oxlint-enable design-system/no-restyle, design-system/no-inline-styles -- end of the SpendMeter tint */
  );
}
