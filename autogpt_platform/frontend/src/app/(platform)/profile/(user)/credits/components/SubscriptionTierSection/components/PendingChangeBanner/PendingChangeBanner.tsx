import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
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

  const isCancellation = pendingTier === "NO_TIER";

  return (
    <div
      role="status"
      aria-live="polite"
      // TODO(tokens): no exact role (border-violet-500 #8B5CF6, text-violet-800 #5B21B6)
      className="flex flex-col gap-2 rounded-md border border-violet-500 bg-accent-surface px-3 py-2 text-violet-800 sm:flex-row sm:items-center sm:justify-between"
    >
      <Text variant="body-tight" as="p" tone="inherit">
        {isCancellation ? (
          <>
            Scheduled to cancel your subscription on{" "}
            <Text variant="body-semibold-tight" as="span" tone="inherit">
              {dateText}
            </Text>
            .
          </>
        ) : (
          <>
            Scheduled to downgrade to{" "}
            <Text variant="body-semibold-tight" as="span" tone="inherit">
              {pendingLabel}
            </Text>{" "}
            on{" "}
            <Text variant="body-semibold-tight" as="span" tone="inherit">
              {dateText}
            </Text>
            .
          </>
        )}
      </Text>
      <Button
        variant="compact-outline"
        size="compact-sm"
        disabled={isBusy}
        onClick={onKeepCurrent}
      >
        {isBusy ? "Cancelling..." : `Keep ${currentLabel}`}
      </Button>
    </div>
  );
}
