import { Button } from "@/components/atoms/Button/Button";
import { approveAllLabel, modeLabel } from "../helpers";

interface Props {
  count: number;
  mode: string | null;
  approveAll: { count: number; busy: boolean; onClick: () => void } | null;
}

export function QueueHeader({ count, mode, approveAll }: Props) {
  return (
    <div className="flex min-h-11 items-center justify-between gap-3 border-b border-zinc-100 bg-zinc-50/60 px-4 py-2">
      <span className="flex items-baseline gap-2 text-sm font-medium text-zinc-900">
        Waiting for you
        <span className="tabular-nums text-zinc-500">{count}</span>
      </span>
      <div className="flex items-center gap-3">
        {modeLabel(mode) && (
          <span className="text-xs text-zinc-500">{modeLabel(mode)}</span>
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
      </div>
    </div>
  );
}
