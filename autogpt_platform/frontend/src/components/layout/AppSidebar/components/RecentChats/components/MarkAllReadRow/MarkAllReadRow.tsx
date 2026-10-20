import { Icon } from "@/components/atoms/Icon/Icon";
import { TickDouble02Icon } from "@hugeicons/core-free-icons";

interface Props {
  count: number;
  onMarkAllRead: () => void;
}

export function MarkAllReadRow({ count, onMarkAllRead }: Props) {
  return (
    <div className="flex items-center justify-between gap-2 px-2 text-xs font-medium text-zinc-500">
      <span className="truncate">
        {count} new {count === 1 ? "reply" : "replies"}
      </span>
      <button
        type="button"
        onClick={onMarkAllRead}
        className="flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 hover:bg-zinc-100 hover:text-zinc-800 max-md:py-2 max-md:text-sm"
      >
        <Icon icon={TickDouble02Icon} className="size-3.5" />
        Mark all as read
      </button>
    </div>
  );
}
