import { Progress } from "@/components/atoms/Progress/Progress";
import { creditsToUsdLabel } from "@/lib/credits";
import { cn } from "@/lib/utils";

interface Props {
  spent: number;
  budget: number;
  muted?: boolean;
}

export function SpendMeter({ spent, budget, muted }: Props) {
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
      className={cn(
        "h-1.5 w-full bg-zinc-100",
        isOverBudget ? "[&>div]:bg-red-400" : "[&>div]:bg-zinc-400",
        muted && "opacity-50",
      )}
    />
  );
}
