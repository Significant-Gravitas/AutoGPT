"use client";

import {
  BubbleChatIcon,
  File02Icon,
  PlayIcon,
} from "@hugeicons/core-free-icons";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertActivity } from "@/app/api/__generated__/models/expertActivity";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { PanelEmptyState } from "@/app/(platform)/copilot/components/ContextPanel/components/PanelEmptyState/PanelEmptyState";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertSummaryCard } from "../ExpertSummaryCard";
import { ActivityChart } from "./components/ActivityChart";
import { LatestOutputs } from "./components/LatestOutputs";
import { OverviewCard } from "./components/OverviewCard";
import { RecentConversations } from "./components/RecentConversations";
import { StatTile } from "./components/StatTile";
import { formatDelta, getBioLead } from "./helpers";
import { useExpertOverview } from "./useExpertOverview";

interface Props {
  expert: Expert;
  activity: ExpertActivity | null;
  isActivityLoading: boolean;
  isActivityError: boolean;
  onChat: () => void;
}

/** How the expert has been doing: headline numbers, the last 30 days of
 *  conversations and runs, and what it most recently said and made. */
export function ExpertOverview({
  expert,
  activity,
  isActivityLoading,
  isActivityError,
  onChat,
}: Props) {
  const overview = useExpertOverview(expert.id, activity);

  return (
    <div className="flex flex-col gap-4">
      {expert.bio ? (
        <Text variant="body-md" tone="secondary">
          {getBioLead(expert.bio)}
        </Text>
      ) : null}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatTile
          label="Conversations"
          icon={BubbleChatIcon}
          value={activity ? overview.conversations.current : null}
          detail={
            activity
              ? formatDelta(overview.conversations.delta)
              : "Last 30 days"
          }
          isLoading={isActivityLoading}
        />
        <StatTile
          label="Workflow runs"
          icon={PlayIcon}
          value={activity ? overview.runs.current : null}
          detail={activity ? formatDelta(overview.runs.delta) : "Last 30 days"}
          isLoading={isActivityLoading}
        />
        <StatTile
          label="Outputs"
          icon={File02Icon}
          value={overview.isOutputsError ? null : overview.outputCount}
          detail={
            overview.isOutputsError
              ? "Unavailable"
              : overview.hasMoreOutputs
                ? "Showing the latest 200"
                : "All time"
          }
          isLoading={overview.isOutputsLoading}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(280px,1fr)]">
        <OverviewCard title="Last 30 days">
          {isActivityLoading ? (
            <Skeleton className="h-56 w-full rounded-lg" />
          ) : isActivityError ? (
            <Text variant="small" tone="muted">
              Couldn&apos;t load the last 30 days.
            </Text>
          ) : overview.hasActivity ? (
            <ActivityChart days={overview.chartDays} />
          ) : (
            <PanelEmptyState
              textSize="body-md"
              tiles={[
                // oxlint-disable-next-line design-system/no-raw-colors -- sky has no role; info blue would change the hue (deferred)
                <span key="a" className="inline-flex text-sky-500">
                  <Icon icon={BubbleChatIcon} size={22} />
                </span>,
                // oxlint-disable-next-line design-system/no-raw-colors -- orange has no role; amber would change the hue (deferred)
                <span key="b" className="inline-flex text-orange-500">
                  <Icon icon={PlayIcon} size={22} />
                </span>,
                <span key="c" className="inline-flex text-brand">
                  <Icon icon={File02Icon} size={22} />
                </span>,
              ]}
              title="Nothing to chart yet"
              description={`Once ${expert.name} starts chatting and running workflows, each day shows up here.`}
            />
          )}
        </OverviewCard>
        <ExpertSummaryCard
          expert={expert}
          activity={activity}
          isActivityLoading={isActivityLoading}
          isActivityError={isActivityError}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <RecentConversations
          expertName={expert.name}
          sessions={overview.sessions}
          isLoading={overview.isSessionsLoading}
          isError={overview.isSessionsError}
          onRetry={overview.retrySessions}
          onChat={onChat}
        />
        <LatestOutputs
          expertName={expert.name}
          outputs={overview.outputs}
          isLoading={overview.isOutputsLoading}
          isError={overview.isOutputsError}
          onRetry={overview.retryOutputs}
        />
      </div>
    </div>
  );
}
