"use client";

import { WorkHistoryIcon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Text } from "@/components/atoms/Text/Text";
import { HomeTile } from "../HomeTile/HomeTile";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
import { WorkGroup } from "./components/WorkGroup";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

/** One card for what the agents did this week, grouped by who did it: each
 *  expert, workflow, or Autopilot with the runs it finished and the files,
 *  integration actions and schedules it produced. */
export function RecentWork({ dashboard, className }: Props) {
  const { briefing } = dashboard;
  const groups = dashboard.recent_work?.groups ?? [];
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
          {briefing.narrative ? (
            <Text
              variant="body"
              tone="secondary"
              className="text-pretty px-4 py-3 leading-5"
            >
              {briefing.narrative}
            </Text>
          ) : null}
          {groups.map((group) => (
            <WorkGroup
              key={
                group.runs?.[0]?.id ?? group.items?.[0]?.id ?? group.actor.name
              }
              group={group}
              timezone={dashboard.timezone}
            />
          ))}
        </div>
      )}
    </HomeTile>
  );
}
