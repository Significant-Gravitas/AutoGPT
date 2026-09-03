"use client";

import {
  CheckmarkCircle02Icon,
  InboxIcon,
  TaskDone01Icon,
} from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatBriefingWindowStart } from "../../helpers";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
import { HomeTileFilter } from "../HomeTileFilter/HomeTileFilter";
import { HomeTile } from "../HomeTile/HomeTile";
import { RecentWorkflowRuns } from "../RecentWorkflowRuns/RecentWorkflowRuns";
import { useRecentWorkflowRuns } from "../RecentWorkflowRuns/useRecentWorkflowRuns";
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
  const recentRuns = useRecentWorkflowRuns();

  return (
    <HomeTile
      className={className}
      icon={TaskDone01Icon}
      title="Your briefing"
      meta={
        <>
          <span className="hidden sm:inline">
            Since{" "}
            {formatBriefingWindowStart(
              briefing.window_started_at,
              dashboard.timezone,
            )}
          </span>
          <span aria-hidden="true" className="hidden text-zinc-300 sm:inline">
            ·
          </span>
          <span className="tabular-nums">
            {briefing.completed_count} completed
          </span>
          {briefing.failed_count > 0 ? (
            <span className="tabular-nums text-rose-600">
              {briefing.failed_count} failed
            </span>
          ) : null}
          {hasFilters ? (
            <HomeTileFilter
              ariaLabelPrefix="Filter briefing outcomes"
              value={selectedFilter}
              options={filterOptions}
              onChange={(value) => selectFilter(value as BriefingFilter)}
            />
          ) : null}
        </>
      }
    >
      {briefing.narrative ? (
        <Text
          variant="body"
          className="text-pretty border-b border-zinc-100 px-4 py-3 text-[13px] leading-5 text-zinc-600"
        >
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
        <div className="divide-y divide-zinc-100">
          {visibleOutcomes.map((outcome) => (
            <OutcomeRow key={outcome.id} outcome={outcome} />
          ))}
        </div>
      )}

      {briefing.routine_count > 0 ? (
        <div className="flex items-center gap-1.5 border-t border-zinc-100 px-4 py-2 text-xs text-zinc-500">
          <Icon
            icon={CheckmarkCircle02Icon}
            size={13}
            className="text-zinc-400"
            aria-hidden="true"
          />
          Plus {briefing.routine_count} routine task
          {briefing.routine_count === 1 ? "" : "s"} completed quietly.
        </div>
      ) : null}

      <RecentWorkflowRuns runs={recentRuns} />
    </HomeTile>
  );
}
