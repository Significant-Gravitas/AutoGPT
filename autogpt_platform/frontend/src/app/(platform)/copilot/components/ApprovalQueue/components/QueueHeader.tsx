import { ArrowUp01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { approveAllLabel, modeLabel } from "../helpers";

interface Props {
  count: number;
  mode: string | null;
  collapsed: boolean;
  onExpand: () => void;
  approveAll: { count: number; busy: boolean; onClick: () => void } | null;
}

export function QueueHeader({
  count,
  mode,
  collapsed,
  onExpand,
  approveAll,
}: Props) {
  const title = (
    <span className="flex items-baseline gap-2 text-sm font-medium text-zinc-900">
      Waiting for you
      <span className="tabular-nums text-zinc-500">{count}</span>
    </span>
  );
  if (collapsed) {
    return (
      <button
        type="button"
        onClick={onExpand}
        className="flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-zinc-300"
      >
        {title}
        <Icon
          icon={ArrowUp01Icon}
          size={14}
          className="text-zinc-400"
          aria-hidden
        />
      </button>
    );
  }
  return (
    <div className="flex min-h-11 items-center justify-between gap-3 border-b border-zinc-100 bg-zinc-50/60 px-4 py-2">
      {title}
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
