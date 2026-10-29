"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

interface Props {
  isPending: boolean;
  isHiringAll: boolean;
  isCardHiring: boolean;
  primaryLabel: string | null;
  secondaryLabel: string | null;
  onPrimary: () => void;
  onSecondary: () => void;
}

// The dark pill hires, the grey line moves on. The line is hidden, not
// disabled, mid-batch: a skip then would advance under a hire still landing.
export function HireStepActions({
  isPending,
  isHiringAll,
  isCardHiring,
  primaryLabel,
  secondaryLabel,
  onPrimary,
  onSecondary,
}: Props) {
  return (
    <div className="flex flex-col items-center gap-2">
      {isPending ? (
        <Skeleton
          className="h-10 w-56 rounded-full"
          data-testid="hire-step-primary-pending"
        />
      ) : isHiringAll || primaryLabel ? (
        <Button
          type="button"
          size="small"
          onClick={onPrimary}
          disabled={isCardHiring}
          loading={isHiringAll}
          className="h-10 w-56"
        >
          {isHiringAll ? "Hiring…" : primaryLabel}
        </Button>
      ) : null}
      {secondaryLabel && !isHiringAll ? (
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={onSecondary}
          disabled={isCardHiring}
          className="text-foreground-tertiary hover:text-foreground-strong"
        >
          {secondaryLabel}
        </Button>
      ) : null}
    </div>
  );
}
