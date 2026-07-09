import {
  useGetV1ListExecutionSchedulesForAUser,
  useListCopilotFollowupSchedules,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { okData } from "@/app/api/helpers";

export type ScheduleItem =
  | { kind: "copilot_turn"; item: CopilotTurnJobInfo }
  | { kind: "graph"; item: GraphExecutionJobInfo };

function nextRunMs(item: { next_run_time?: string | null }): number {
  if (!item.next_run_time) return Number.POSITIVE_INFINITY;
  const t = new Date(item.next_run_time).valueOf();
  return Number.isNaN(t) ? Number.POSITIVE_INFINITY : t;
}

export function useFollowupsPage() {
  // Followups (copilot_turn) — the scheduled-message side of the
  // feature.  Stored via ``schedule_followup`` MCP tool.
  const copilotQuery = useListCopilotFollowupSchedules({
    query: { select: (res) => okData(res) ?? [] },
  });
  // Graph schedules — recurring agent runs created via the agent
  // builder.  Same scheduler, different ``kind`` discriminator.  This
  // page now unifies both so users see ALL their pending automated
  // work in one place (briefing pill, this list, single Delete flow).
  const graphQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [] },
  });

  const copilotItems: ScheduleItem[] = (copilotQuery.data ?? []).map(
    (item) => ({ kind: "copilot_turn" as const, item }),
  );
  const graphItems: ScheduleItem[] = (graphQuery.data ?? []).map((item) => ({
    kind: "graph" as const,
    item,
  }));

  // Merge + sort by next_run_time ascending so the soonest-firing
  // schedule lands at the top.  Items with no ``next_run_time`` (e.g.
  // a one-shot graph schedule that has already fired but hasn't been
  // cleaned up yet) sink to the bottom.
  const schedules = [...copilotItems, ...graphItems].sort(
    (a, b) => nextRunMs(a.item) - nextRunMs(b.item),
  );

  return {
    // Backwards-compat alias — ``followups`` was the only-copilot
    // collection name before unification.  Existing tests still
    // reference it.
    followups: copilotQuery.data ?? [],
    schedules,
    isLoading: copilotQuery.isLoading || graphQuery.isLoading,
    error: copilotQuery.error ?? graphQuery.error,
  };
}
