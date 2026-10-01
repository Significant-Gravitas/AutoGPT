import { Calendar03Icon } from "@hugeicons/core-free-icons";
import type { HomeUpcomingTask } from "@/app/api/__generated__/models/homeUpcomingTask";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { WorkflowAvatar } from "@/components/molecules/WorkflowAvatar/WorkflowAvatar";
import { formatUntil } from "../helpers";

interface Props {
  item: HomeUpcomingTask;
}

export function UpcomingRow({ item }: Props) {
  return (
    <div className="relative flex items-center gap-3 py-2 pl-[3.75rem] pr-4">
      <span className="absolute left-4 top-1/2 z-10 flex size-9 -translate-y-1/2 items-center justify-center rounded-full bg-white">
        <TaskMarker item={item} />
      </span>
      <div className="min-w-0 flex-1">
        <Text
          variant="body-medium"
          tone="primary"
          className="truncate leading-5"
        >
          <span>{item.title}</span>
          {item.kind === "agent" ? (
            <span className="font-normal text-zinc-500"> workflow</span>
          ) : null}
        </Text>
        <Text variant="body" tone="muted" className="truncate">
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

/** The expert who will do the work is the marker; a workflow shows its own
 *  picture, and the glyph only stands in for a follow-up. */
function TaskMarker({ item }: Props) {
  if (item.expert) {
    return (
      <ExpertAvatar
        name={item.expert.name}
        avatarUrl={item.expert.avatar_url}
        size={36}
      />
    );
  }
  if (item.kind === "agent") {
    return (
      <WorkflowAvatar name={item.title} imageUrl={item.image_url} size={36} />
    );
  }
  return (
    <span className="flex size-9 items-center justify-center rounded-full bg-zinc-100 text-zinc-500 ring-2 ring-white">
      <Icon icon={Calendar03Icon} size={15} aria-hidden="true" />
    </span>
  );
}
