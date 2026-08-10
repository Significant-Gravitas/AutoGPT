import { Calendar03Icon, Clock01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeActiveTask } from "@/app/api/__generated__/models/homeActiveTask";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import type { HomeUpcomingTask } from "@/app/api/__generated__/models/homeUpcomingTask";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { HomeTile } from "../HomeTile/HomeTile";
import { formatRunningFor, formatUntil } from "./helpers";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

export function NowNext({ dashboard, className }: Props) {
  return (
    <HomeTile
      className={className}
      contentClassName="flex flex-col gap-4"
      title={
        <div className="flex items-center gap-2">
          <Icon
            icon={Calendar03Icon}
            size={18}
            className="text-zinc-500"
            aria-hidden="true"
          />
          <Text variant="h5" className="text-zinc-950">
            Now &amp; next
          </Text>
        </div>
      }
      header={
        <Text variant="large" className="text-zinc-600">
          Live work and the next scheduled handoffs.
        </Text>
      }
    >
      {dashboard.active_tasks.length > 0 ? (
        <div>
          <Text variant="small" className="font-medium text-zinc-500">
            Working now
          </Text>
          <div className="relative mt-1 before:absolute before:bottom-4 before:left-4 before:top-4 before:w-px before:bg-zinc-200">
            {dashboard.active_tasks.map((item) => (
              <ActiveRow key={item.id} item={item} />
            ))}
          </div>
        </div>
      ) : null}

      <div>
        <Text variant="small" className="font-medium text-zinc-500">
          Coming up
        </Text>
        {dashboard.upcoming_tasks.length === 0 ? (
          <div className="py-6 text-center">
            <Text variant="small" className="text-pretty text-zinc-500">
              Nothing is scheduled. Your agents are ready when you are.
            </Text>
          </div>
        ) : (
          <div className="relative mt-1 before:absolute before:bottom-4 before:left-4 before:top-4 before:w-px before:bg-zinc-200">
            {dashboard.upcoming_tasks.map((item) => (
              <UpcomingRow key={item.id} item={item} />
            ))}
          </div>
        )}
      </div>
    </HomeTile>
  );
}

function ActiveRow({ item }: { item: HomeActiveTask }) {
  const content = (
    <>
      <span className="absolute left-0 top-1/2 z-10 flex size-8 -translate-y-1/2 items-center justify-center rounded-full bg-white">
        {item.expert ? (
          <ExpertAvatar
            name={item.expert.name}
            avatarUrl={item.expert.avatar_url}
            size={32}
          />
        ) : (
          <span className="size-2.5 rounded-full bg-primary" />
        )}
      </span>
      <div className="min-w-0 flex-1">
        <Text variant="body-medium" className="truncate text-zinc-900">
          {item.title}
        </Text>
        <Text variant="small" className="truncate text-zinc-500">
          {item.status === "queued"
            ? "Queued"
            : (formatRunningFor(item.started_at) ?? "Running now")}
        </Text>
      </div>
    </>
  );
  const classes = "relative flex items-center gap-3 rounded-xl py-3 pl-12 pr-2";

  if (!item.link) return <div className={classes}>{content}</div>;
  return (
    <Link
      href={item.link}
      className={cn(
        classes,
        "transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400",
      )}
    >
      {content}
    </Link>
  );
}

function UpcomingRow({ item }: { item: HomeUpcomingTask }) {
  return (
    <div className="relative flex items-center gap-3 rounded-xl py-3 pl-12 pr-2">
      <span className="absolute left-0 top-1/2 z-10 flex size-8 -translate-y-1/2 items-center justify-center rounded-full bg-zinc-100 text-zinc-500">
        <Icon
          icon={item.kind === "followup" ? Calendar03Icon : Clock01Icon}
          size={15}
          aria-hidden="true"
        />
      </span>
      <div className="min-w-0 flex-1">
        <Text variant="body-medium" className="truncate text-zinc-900">
          {item.title}
        </Text>
        <Text variant="small" className="truncate text-zinc-500">
          {item.expert?.name ??
            (item.kind === "followup" ? "Follow-up" : "Scheduled task")}
        </Text>
      </div>
      <Text
        variant="small"
        className="shrink-0 font-medium tabular-nums text-zinc-700"
      >
        {formatUntil(item.next_run_time)}
      </Text>
    </div>
  );
}
