"use client";

import { WorkHistoryIcon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import type { HomeRecentWorkGroup } from "@/app/api/__generated__/models/homeRecentWorkGroup";
import { Text } from "@/components/atoms/Text/Text";
import { HomeSectionLabel } from "../HomeSectionLabel/HomeSectionLabel";
import { HomeTile } from "../HomeTile/HomeTile";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
import { BriefingByline } from "./components/BriefingByline";
import { WorkGroup } from "./components/WorkGroup";
import { splitGroupsBySection } from "./helpers";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

/** One card for what the agents did this week: the team (each expert and
 *  Autopilot) first, then the workflows that ran on their own in a section
 *  of their own below. */
export function RecentWork({ dashboard, className }: Props) {
  const { briefing } = dashboard;
  const groups = dashboard.recent_work?.groups ?? [];
  const { team, workflows } = splitGroupsBySection(groups);
  const completed = dashboard.recent_work?.completed_count ?? 0;
  const failed = dashboard.recent_work?.failed_count ?? 0;
  const isEmpty = groups.length === 0 && !briefing.narrative;

  return (
    <HomeTile
      className={className}
      icon={WorkHistoryIcon}
      title="Recent work"
      meta={
        <>
          <Text
            variant="small"
            as="span"
            tone="muted"
            className="hidden sm:inline"
          >
            This week
          </Text>
          <span aria-hidden="true" className="hidden text-zinc-300 sm:inline">
            ·
          </span>
          <Text variant="small" as="span" tone="muted" className="tabular-nums">
            {completed} completed
          </Text>
          {failed > 0 ? (
            <Text
              variant="small"
              as="span"
              className="tabular-nums text-rose-600"
            >
              {failed} failed
            </Text>
          ) : null}
        </>
      }
    >
      {isEmpty ? (
        <HomeTileEmpty
          title="Nothing to show yet"
          description="Runs, files, integrations and schedules from your experts and workflows will appear here."
        />
      ) : (
        <div className="divide-y divide-zinc-200">
          {briefing.narrative ? <BriefingByline briefing={briefing} /> : null}
          {team.map((group) => (
            <WorkGroup
              key={groupKey(group)}
              group={group}
              timezone={dashboard.timezone}
            />
          ))}
          {workflows.length > 0 ? (
            <div className="pt-2">
              <HomeSectionLabel>Workflows</HomeSectionLabel>
              <div className="divide-y divide-zinc-200 border-t border-zinc-200">
                {workflows.map((group) => (
                  <WorkGroup
                    key={groupKey(group)}
                    group={group}
                    timezone={dashboard.timezone}
                  />
                ))}
              </div>
            </div>
          ) : null}
        </div>
      )}
    </HomeTile>
  );
}

function groupKey(group: HomeRecentWorkGroup) {
  return group.runs?.[0]?.id ?? group.items?.[0]?.id ?? group.actor.name;
}
