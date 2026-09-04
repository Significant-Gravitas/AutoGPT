import { creditsToUsdLabel } from "@/lib/credits";
import { cn } from "@/lib/utils";

const SEGMENT_COUNT = 36;

interface Props {
  spent: number;
  budget: number;
  className?: string;
}

export function SpendMeter({ spent, budget, className }: Props) {
  const ratio = budget > 0 ? Math.min(Math.max(spent / budget, 0), 1) : 0;
  const filledCount = Math.round(ratio * SEGMENT_COUNT);
  const clampedSpent = Math.min(Math.max(spent, 0), Math.max(budget, 0));
  const valueText = `${creditsToUsdLabel(spent)} of ${creditsToUsdLabel(budget)} spent this week${
    spent > budget ? " (over budget)" : ""
  }`;

  return (
    <div
      role="progressbar"
      aria-label="Weekly spend"
      aria-valuenow={clampedSpent}
      aria-valuemin={0}
      aria-valuemax={budget}
      aria-valuetext={valueText}
      className={cn("flex h-4 w-full items-stretch gap-[3px]", className)}
    >
      {Array.from({ length: SEGMENT_COUNT }, (_, index) => (
        <span
          key={index}
          className={cn(
            "flex-1 rounded-[1px] transition-colors duration-300",
            index < filledCount ? "bg-zinc-800" : "bg-zinc-200",
          )}
        />
      ))}
    </div>
  );
}
