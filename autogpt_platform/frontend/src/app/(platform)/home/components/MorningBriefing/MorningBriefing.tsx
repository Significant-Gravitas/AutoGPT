"use client";

import {
  CheckmarkCircle02Icon,
  InboxIcon,
  TaskDone01Icon,
} from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { useTrackFunnelViewOnce } from "@/services/experts/use-track-funnel-view-once";
import { formatBriefingWindowStart } from "../../helpers";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
import { HomeTileFilter } from "../HomeTileFilter/HomeTileFilter";
import { HomeTile } from "../HomeTile/HomeTile";
import { OutcomeRow } from "./components/OutcomeRow";
import { type BriefingFilter, useMorningBriefing } from "./useMorningBriefing";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

export function MorningBriefing({ dashboard, className }: Props) {
  const { briefing } = dashboard;
  const {
    filterOptions,
    hasFilters,
    selectedFilter,
    selectFilter,
    visibleOutcomes,
  } = useMorningBriefing({ outcomes: briefing.outcomes });

  useTrackFunnelViewOnce("briefing_opened");

  return (
    <HomeTile
      className={className}
      contentClassName="flex flex-col gap-4"
      surfaceClassName="py-4 sm:py-4"
      title={
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <Icon
              icon={TaskDone01Icon}
              size={18}
              className="text-zinc-500"
              aria-hidden="true"
            />
            <Text variant="h5" className="text-zinc-950">
              Your briefing
            </Text>
          </div>
          <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
            <div className="flex items-center gap-3 text-xs font-medium tabular-nums text-zinc-500">
              <span>{briefing.completed_count} completed</span>
              {briefing.failed_count > 0 ? (
                <span className="text-rose-700">
                  {briefing.failed_count} failed
                </span>
              ) : null}
            </div>
            {hasFilters ? (
              <HomeTileFilter
                ariaLabelPrefix="Filter briefing outcomes"
                value={selectedFilter}
                options={filterOptions}
                onChange={(value) => selectFilter(value as BriefingFilter)}
              />
            ) : null}
          </div>
        </div>
      }
      header={
        <Text variant="large" className="text-zinc-600">
          The outcomes worth knowing since{" "}
          {formatBriefingWindowStart(
            briefing.window_started_at,
            dashboard.timezone,
          )}
          .
        </Text>
      }
    >
      {briefing.narrative ? (
        <Text variant="large" className="text-zinc-700">
          {briefing.narrative}
        </Text>
      ) : null}

      {briefing.outcomes.length === 0 ? (
        <HomeTileEmpty
          icon={InboxIcon}
          title="No new outcomes yet"
          description="Completed work and useful exceptions will appear here."
        />
      ) : (
        <div className="-mx-4 divide-y divide-zinc-100 sm:-mx-5">
          {visibleOutcomes.map((outcome) => (
            <OutcomeRow key={outcome.id} outcome={outcome} />
          ))}
        </div>
      )}

      {briefing.routine_count > 0 ? (
        <div className="inline-flex items-center gap-1.5 self-end rounded-full border border-purple-500 bg-purple-100 px-2.5 py-1 text-sm font-medium text-purple-600">
          <Icon icon={CheckmarkCircle02Icon} size={15} aria-hidden="true" />
          Plus {briefing.routine_count} routine task
          {briefing.routine_count === 1 ? "" : "s"} completed quietly.
        </div>
      ) : null}
    </HomeTile>
  );
}
