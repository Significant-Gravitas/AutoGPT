import { ditherColorsFor } from "@/app/(platform)/raise/components/ColorStep/helpers";
import { creditsToUsdLabel } from "@/lib/credits";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

const SEGMENT_COUNT = 28;
/** Segments light up left to right, so the bar fills up instead of blinking on. */
const SEGMENT_DELAY_MS = 25;

interface Props {
  spent: number;
  budget: number;
  color?: string | null;
  muted?: boolean;
}

export function SpendMeter({ spent, budget, color, muted }: Props) {
  const [hasEntered, setHasEntered] = useState(false);
  const ratio = budget > 0 ? Math.min(Math.max(spent / budget, 0), 1) : 0;
  const filledCount = hasEntered ? Math.round(ratio * SEGMENT_COUNT) : 0;

  useEffect(() => setHasEntered(true), []);
  // The dither ramp's second stop is the -300 shade the expert was picked in.
  const fillColor = ditherColorsFor(color ?? null)?.[1];
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
        "flex h-4 w-full items-stretch gap-1",
        muted && "opacity-50",
      )}
    >
      {Array.from({ length: SEGMENT_COUNT }, (_, index) => {
        const isFilled = index < filledCount;

        return (
          <span
            key={index}
            style={{
              transitionDelay: `${index * SEGMENT_DELAY_MS}ms`,
              backgroundColor: isFilled && fillColor ? fillColor : undefined,
            }}
            className={cn(
              "flex-1 rounded-[1px] transition-colors duration-300",
              isFilled && !fillColor ? "bg-zinc-800" : "bg-zinc-200",
            )}
          />
        );
      })}
    </div>
  );
}
