import { Calendar03Icon, Clock01Icon } from "@hugeicons/core-free-icons";
import type { HomeUpcomingTask } from "@/app/api/__generated__/models/homeUpcomingTask";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { formatUntil } from "../helpers";

interface Props {
  item: HomeUpcomingTask;
}

export function UpcomingRow({ item }: Props) {
  return (
    <div className="relative flex items-center gap-3 py-2 pl-[3.75rem] pr-4">
      {/* The expert who will do the work is the marker; the glyph only
          stands in when no expert owns the task. */}
      <span className="absolute left-4 top-1/2 z-10 flex size-9 -translate-y-1/2 items-center justify-center rounded-full bg-white">
        {item.expert ? (
          <ExpertAvatar
            name={item.expert.name}
            avatarUrl={item.expert.avatar_url}
            size={36}
          />
        ) : (
          <span className="flex size-9 items-center justify-center rounded-full bg-zinc-100 text-zinc-500 ring-2 ring-white">
            <Icon
              icon={item.kind === "followup" ? Calendar03Icon : Clock01Icon}
              size={15}
              aria-hidden="true"
            />
          </span>
        )}
      </span>
      <div className="min-w-0 flex-1">
        <Text
          variant="body-medium"
          className="truncate text-[13px] leading-5 text-zinc-900"
        >
          {item.title}
        </Text>
        <Text variant="body" className="truncate text-zinc-500">
          {item.expert?.name ??
            (item.kind === "followup" ? "Follow-up" : "Scheduled task")}
          <span aria-hidden="true"> · </span>
          <span className="tabular-nums">
            {formatUntil(item.next_run_time)}
          </span>
        </Text>
      </div>
    </div>
  );
}
