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
              "--meter-track": `color-mix(in srgb, ${color} 45%, rgb(var(--color-card)))`,
              "--meter-fill": `color-mix(in srgb, ${color} 60%, rgb(var(--color-primary)))`,
            } as CSSProperties)
          : undefined
      }
      className={cn(
        "h-1.5 w-full bg-surface-hover",
        color && "h-2 ![background-color:var(--meter-track)]",
        isOverBudget
          ? // TODO(tokens): no exact role: [&>div]:bg-red-400 #F26969
            "[&>div]:bg-red-400"
          : color
            ? "[&>div]:![background-color:var(--meter-fill)]"
            : // TODO(tokens): no fitting role for [&>div]:bg-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour)
              "[&>div]:bg-zinc-400",
        muted && "opacity-50",
      )}
    />
  );
}
