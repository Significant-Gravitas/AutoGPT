import type { ReactNode } from "react";

interface Props {
  spent: number;
  threshold: number;
  children: ReactNode;
}

const SIZE = 44;
const RADIUS = 19.5;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

// One full lap is the check-in amount; spend past it laps the ring again.
export function SpendRing({ spent, threshold, children }: Props) {
  const ratio = threshold > 0 ? spent / threshold : 1;
  const base = Math.min(ratio, 1) * CIRCUMFERENCE;
  const extra = Math.min(Math.max(ratio - 1, 0), 1) * CIRCUMFERENCE;

  return (
    <div
      role="img"
      aria-label={`${spent} of ${threshold} credits`}
      className="relative grid size-11 shrink-0 place-items-center"
    >
      <svg
        width={SIZE}
        height={SIZE}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="absolute inset-0 -rotate-90"
        aria-hidden="true"
      >
        <Arc className="stroke-border-base" length={CIRCUMFERENCE} />
        <Arc className="stroke-foreground-strong" length={base} />
        {extra > 0 && (
          <>
            <Arc className="stroke-surface" length={extra} width={5} />
            <Arc className="stroke-foreground-subtle" length={extra} />
          </>
        )}
      </svg>
      {children}
    </div>
  );
}

interface ArcProps {
  className: string;
  length: number;
  width?: number;
}

function Arc({ className, length, width = 2.5 }: ArcProps) {
  return (
    <circle
      cx={SIZE / 2}
      cy={SIZE / 2}
      r={RADIUS}
      fill="none"
      strokeWidth={width}
      strokeLinecap={length < CIRCUMFERENCE ? "round" : undefined}
      strokeDasharray={`${length} ${CIRCUMFERENCE}`}
      className={className}
    />
  );
}
