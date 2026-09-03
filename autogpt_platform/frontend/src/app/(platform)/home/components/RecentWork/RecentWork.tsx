"use client";

import {
  CheckmarkCircle02Icon,
  WorkHistoryIcon,
} from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatBriefingWindowStart } from "../../helpers";
import { HomeSectionLabel } from "../HomeSectionLabel/HomeSectionLabel";
import { HomeTile } from "../HomeTile/HomeTile";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
import { HomeTileFilter } from "../HomeTileFilter/HomeTileFilter";
import { ListeningAgents } from "../ListeningAgents/ListeningAgents";
import { useListeningAgents } from "../ListeningAgents/useListeningAgents";
import { OutcomeRow } from "./components/OutcomeRow";
import { WorkGroup } from "./components/WorkGroup";
import { type OutcomeFilter, useOutcomeFilter } from "./useOutcomeFilter";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

/** One card for what the agents did: the briefing's run outcomes, then the
 *  durable things those runs produced, then a line for agents still waiting
 *  on a trigger. The two feeds describe the same day from two angles, so
 *  they share a header and one empty state. */
export function RecentWork({ dashboard, className }: Props) {
  const { briefing } = dashboard;
  const groups = dashboard.recent_work?.groups ?? [];
  const {
    filterOptions,
    hasFilters,
    selectedFilter,
    selectFilter,
    visibleOutcomes,
  } = useOutcomeFilter({ outcomes: briefing.outcomes });
  const listeningAgents = useListeningAgents();
  const hasOutcomes = briefing.outcomes.length > 0;
  const hasGroups = groups.length > 0;
  const isEmpty = !hasOutcomes && !hasGroups && !briefing.narrative;

  return (
    <HomeTile
      className={className}
      icon={WorkHistoryIcon}
      title="Recent work"
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
              onChange={(value) => selectFilter(value as OutcomeFilter)}
            />
          ) : null}
        </>
      }
    >
      {isEmpty ? (
        <HomeTileEmpty
          title="Nothing to show yet"
          description="Completed runs, files, integrations and schedules will appear here."
        />
      ) : (
        <div className="divide-y divide-zinc-100">
          {briefing.narrative ? (
            <Text
              variant="body"
              className="text-pretty px-4 py-3 text-[13px] leading-5 text-zinc-600"
            >
              {briefing.narrative}
            </Text>
          ) : null}

          {hasOutcomes ? (
            <section aria-label="Outcomes" className="pb-1">
              <HomeSectionLabel>Outcomes</HomeSectionLabel>
              <div className="divide-y divide-zinc-100">
                {visibleOutcomes.map((outcome) => (
                  <OutcomeRow key={outcome.id} outcome={outcome} />
                ))}
              </div>
              {briefing.routine_count > 0 ? (
                <div className="flex items-center gap-1.5 px-4 py-2 text-xs text-zinc-500">
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
            </section>
          ) : null}

          {hasGroups ? (
            <section aria-label="Delivered">
              <HomeSectionLabel>Delivered this week</HomeSectionLabel>
              <div className="divide-y divide-zinc-100">
                {groups.map((group) => (
                  <WorkGroup
                    key={group.items[0]?.id ?? group.actor.name}
                    group={group}
                    timezone={dashboard.timezone}
                  />
                ))}
              </div>
            </section>
          ) : null}
        </div>
      )}

      <ListeningAgents agents={listeningAgents} />
    </HomeTile>
  );
}
