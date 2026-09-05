"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { Text } from "@/components/atoms/Text/Text";
import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import {
  SECTION_INSET_CLASS,
  TEAM_GRID_CLASS,
  getAutopilotSummary,
} from "../../helpers";
import { AutopilotCard } from "../AutopilotCard";
import { ExpertTeamCardSkeleton } from "../ExpertTeamCardSkeleton";
import { TeamRosterToolbar } from "./TeamRosterToolbar";
import { useTeamRosterView } from "./useTeamRosterView";

interface Props {
  isLoading: boolean;
  experts: Expert[];
  schedulesForExpert: (expert: Expert) => GraphExecutionJobInfo[];
  renderCard: (expert: Expert) => ReactNode;
}

export function TeamRoster({
  isLoading,
  experts,
  schedulesForExpert,
  renderCard,
}: Props) {
  const { query, setQuery, filter, setFilter, isNarrowed, visibleExperts } =
    useTeamRosterView({ experts, schedulesForExpert });

  // Autopilot reports on the whole team, so its summary ignores the toolbar.
  const summary = getAutopilotSummary({ experts, schedulesForExpert });
  const autopilot = (
    <AutopilotCard
      skillCount={summary.skillCount}
      scheduleCount={summary.scheduleCount}
      workflowCount={summary.workflowCount}
    />
  );

  return (
    <section aria-label="Experts" className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <Text variant="h5">Experts</Text>
        <TeamRosterToolbar
          query={query}
          onQueryChange={setQuery}
          filter={filter}
          onFilterChange={setFilter}
        />
      </div>

      {isLoading ? (
        <div className={TEAM_GRID_CLASS}>
          {autopilot}
          {[0, 1, 2].map((index) => (
            <ExpertTeamCardSkeleton key={index} />
          ))}
        </div>
      ) : (
        <div className={TEAM_GRID_CLASS}>
          {/* Autopilot is pinned rather than filtered — it is always on the
              team, so it stays put unless the roster is being narrowed. */}
          {isNarrowed ? null : autopilot}
          {visibleExperts.map(renderCard)}
        </div>
      )}

      {!isLoading && isNarrowed && visibleExperts.length === 0 ? (
        <Text
          variant="body"
          className={cn("text-zinc-500", SECTION_INSET_CLASS)}
        >
          No experts match that search or filter.
        </Text>
      ) : null}
    </section>
  );
}
