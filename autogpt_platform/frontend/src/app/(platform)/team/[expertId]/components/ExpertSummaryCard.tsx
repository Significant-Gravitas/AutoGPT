import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertActivity } from "@/app/api/__generated__/models/expertActivity";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { FireIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertActivityGraph } from "./ExpertActivityGraph/ExpertActivityGraph";
import {
  getActivityStreak,
  getActivitySummary,
  getThreeMonthActivityDays,
  getYearActivityDays,
} from "./ExpertActivityGraph/helpers";

interface Props {
  expert: Expert;
  activity: ExpertActivity | null;
  isActivityLoading: boolean;
  isActivityError: boolean;
}

export function ExpertSummaryCard({
  expert,
  activity,
  isActivityLoading,
  isActivityError,
}: Props) {
  const activityDays = activity ? getYearActivityDays(activity.days) : null;
  const visibleActivityDays = activityDays
    ? getThreeMonthActivityDays(activityDays)
    : null;
  const summary = visibleActivityDays
    ? getActivitySummary(visibleActivityDays)
    : null;
  const streak = activityDays ? getActivityStreak(activityDays) : null;

  return (
    <aside
      aria-label={`${expert.name} at a glance`}
      className="flex flex-col gap-4 self-start"
    >
      <section
        aria-label={`${expert.name} activity`}
        className="flex flex-col gap-2 rounded-xl border border-zinc-200 bg-white p-4"
      >
        <div className="flex items-center justify-between gap-2">
          <Text variant="large-medium" className="text-base text-zinc-700">
            Activity
          </Text>
          {summary ? <ActivityStatus isActive={summary.isActive} /> : null}
        </div>
        {isActivityLoading ? (
          <Skeleton className="h-[5.25rem] w-full rounded-lg" />
        ) : isActivityError ? (
          <Text variant="small" className="text-zinc-500">
            Activity unavailable
          </Text>
        ) : visibleActivityDays && summary ? (
          <>
            <ExpertActivityGraph
              days={visibleActivityDays}
              color={expert.color}
            />
            <Text variant="small" className="text-zinc-500">
              {summary.totalsLabel} · {summary.rangeLabel}
            </Text>
          </>
        ) : null}
      </section>

      <section
        aria-label={`${expert.name} activity streak`}
        className="flex flex-col rounded-xl border border-zinc-200 bg-white p-4"
      >
        {isActivityLoading ? (
          <Skeleton className="h-16 w-full rounded-lg" />
        ) : isActivityError ? (
          <Text variant="small" className="text-zinc-500">
            Streak unavailable
          </Text>
        ) : (
          <div className="flex items-center justify-center gap-3 text-center">
            <Icon
              icon={FireIcon}
              size={40}
              className="shrink-0 text-zinc-400"
            />
            <div className="min-w-0">
              <Text variant="h4" className="tabular-nums text-zinc-900">
                {streak ?? 0}
              </Text>
              <Text variant="small" className="text-sm text-zinc-500">
                day streak
              </Text>
            </div>
          </div>
        )}
      </section>
    </aside>
  );
}

function ActivityStatus({ isActive }: { isActive: boolean }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        aria-hidden
        className={cn(
          "size-2 rounded-full",
          isActive ? "bg-emerald-500" : "bg-zinc-300",
        )}
      />
      <Text variant="small" className="text-zinc-500">
        {isActive ? "Active this week" : "Quiet lately"}
      </Text>
    </span>
  );
}
