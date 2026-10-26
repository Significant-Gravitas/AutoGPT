import { Button } from "@/components/atoms/Button/Button";
import { approveAllLabel, modeLabel } from "../helpers";

interface Props {
  count: number;
  mode: string | null;
  approveAll: { count: number; busy: boolean; onClick: () => void } | null;
  onReviewAll: (() => void) | null;
  // Collapsed after Close and continue: the header alone, and a way back.
  onShow: (() => void) | null;
}

export function QueueHeader({
  count,
  mode,
  approveAll,
  onReviewAll,
  onShow,
}: Props) {
  return (
    <div className="flex min-h-11 flex-wrap items-center justify-between gap-x-3 gap-y-2 border-b border-separator bg-surface-muted/60 px-4 py-2">
      <span className="flex items-baseline gap-2 text-sm font-medium text-foreground-strong">
        Waiting for you
        <span className="tabular-nums text-foreground-tertiary">{count}</span>
      </span>
      <div className="flex items-center gap-3">
        {modeLabel(mode) && (
          <span className="text-xs text-foreground-tertiary">
            {modeLabel(mode)}
          </span>
        )}
        {onShow ? (
          <Button
            type="button"
            size="small"
            variant="secondary"
            className="min-w-0"
            aria-label="Show waiting cards"
            aria-expanded={false}
            // Close and continue removed the button that had focus.
            autoFocus
            onClick={onShow}
          >
            Show
          </Button>
        ) : (
          <>
            {onReviewAll && (
              <Button
                type="button"
                size="small"
                variant="secondary"
                className="min-w-0"
                onClick={onReviewAll}
              >
                Review all
              </Button>
            )}
            {approveAll && (
              <Button
                size="small"
                variant="secondary"
                className="min-w-0"
                disabled={approveAll.busy}
                onClick={approveAll.onClick}
              >
                {approveAllLabel(approveAll.count)}
              </Button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
