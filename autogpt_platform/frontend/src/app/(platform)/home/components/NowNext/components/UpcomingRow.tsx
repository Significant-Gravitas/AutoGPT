import { Calendar03Icon, Clock01Icon } from "@hugeicons/core-free-icons";
import type { HomeUpcomingTask } from "@/app/api/__generated__/models/homeUpcomingTask";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatUntil } from "../helpers";

interface Props {
  item: HomeUpcomingTask;
}

export function UpcomingRow({ item }: Props) {
  return (
    <div className="relative flex items-center gap-3 py-2 pl-[3.75rem] pr-4">
      <span className="absolute left-4 top-1/2 z-10 flex size-9 -translate-y-1/2 items-center justify-center rounded-full bg-zinc-100 text-zinc-500 ring-2 ring-white">
        <Icon
          icon={item.kind === "followup" ? Calendar03Icon : Clock01Icon}
          size={15}
          aria-hidden="true"
        />
      </span>
      <div className="min-w-0 flex-1">
        <Text
          variant="body-medium"
          className="truncate text-[13px] leading-5 text-zinc-900"
        >
          {item.title}
        </Text>
        <Text variant="small" className="truncate text-[11px] text-zinc-500">
          {item.expert?.name ??
            (item.kind === "followup" ? "Follow-up" : "Scheduled task")}
        </Text>
      </div>
      <Text
        variant="small"
        className="shrink-0 text-[11px] font-medium tabular-nums text-zinc-600"
      >
        {formatUntil(item.next_run_time)}
      </Text>
    </div>
  );
}
