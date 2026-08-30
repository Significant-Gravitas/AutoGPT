import { creditsToUsdLabel } from "@/lib/credits";
import { cn } from "@/lib/utils";

const SEGMENT_COUNT = 36;

interface Props {
  spent: number;
  budget: number;
  muted?: boolean;
}

export function SpendMeter({ spent, budget, muted }: Props) {
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
      className={cn(
        // Segments are `flex-1`, so the gap is what sets their width — a wider
        // gap thins each tick without changing how many there are.
        "flex h-4 w-full items-stretch gap-[5px]",
        muted && "opacity-50",
      )}
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
