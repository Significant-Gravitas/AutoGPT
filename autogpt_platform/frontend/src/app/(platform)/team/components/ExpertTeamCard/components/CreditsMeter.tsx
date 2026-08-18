import { cn } from "@/lib/utils";

const SEGMENT_COUNT = 36;

interface Props {
  spent: number;
  budget: number;
  muted?: boolean;
}

export function CreditsMeter({ spent, budget, muted }: Props) {
  const ratio = budget > 0 ? Math.min(Math.max(spent / budget, 0), 1) : 0;
  const filledCount = Math.round(ratio * SEGMENT_COUNT);

  return (
    <div
      role="progressbar"
      aria-valuenow={spent}
      aria-valuemin={0}
      aria-valuemax={budget}
      className={cn(
        "flex h-4 w-full items-stretch gap-[3px]",
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
