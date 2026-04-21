import { Button } from "@/components/ui/button";
import { formatPendingDate, getTierLabel } from "../../helpers";

interface Props {
  currentTier: string;
  pendingTier: string;
  pendingEffectiveAt: Date | string | null | undefined;
  onKeepCurrent: () => void;
  isBusy: boolean;
}

export function PendingChangeBanner({
  currentTier,
  pendingTier,
  pendingEffectiveAt,
  onKeepCurrent,
  isBusy,
}: Props) {
  // Backend invariant: pending_tier_effective_at is always populated when
  // pending_tier is set. Bail early if the date is missing so the sentence
  // always reads with a date instead of a null-fallback branch.
  if (!pendingEffectiveAt) return null;

  const pendingLabel = getTierLabel(pendingTier);
  const currentLabel = getTierLabel(currentTier);
  const dateText = formatPendingDate(pendingEffectiveAt);

  const isCancellation = pendingTier === "FREE";

  return (
    <div
      role="status"
      aria-live="polite"
      className="flex flex-col gap-2 rounded-md border border-violet-500 bg-violet-50 px-3 py-2 text-sm text-violet-800 sm:flex-row sm:items-center sm:justify-between"
    >
      <p>
        {isCancellation ? (
          <>
            Scheduled to cancel your subscription on{" "}
            <span className="font-semibold">{dateText}</span>.
          </>
        ) : (
          <>
            Scheduled to downgrade to{" "}
            <span className="font-semibold">{pendingLabel}</span> on{" "}
            <span className="font-semibold">{dateText}</span>.
          </>
        )}
      </p>
      <Button
        variant="outline"
        size="sm"
        disabled={isBusy}
        onClick={onKeepCurrent}
      >
        {isBusy ? "Cancelling..." : `Keep ${currentLabel}`}
      </Button>
    </div>
  );
}
