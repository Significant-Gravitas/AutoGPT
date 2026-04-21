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
  const pendingLabel = getTierLabel(pendingTier);
  const currentLabel = getTierLabel(currentTier);
  const dateText = pendingEffectiveAt
    ? formatPendingDate(pendingEffectiveAt)
    : null;

  return (
    <div
      role="status"
      className="flex flex-col gap-2 rounded-md border border-violet-500 bg-violet-50 px-3 py-2 text-sm text-violet-800 dark:bg-violet-900/20 dark:text-violet-200 sm:flex-row sm:items-center sm:justify-between"
    >
      <p>
        Scheduled to {pendingTier === "FREE" ? "cancel" : "downgrade"} to{" "}
        <span className="font-semibold">{pendingLabel}</span>
        {dateText ? (
          <>
            {" "}
            on <span className="font-semibold">{dateText}</span>
          </>
        ) : null}
        .
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
