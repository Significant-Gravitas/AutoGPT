import {
  useGetV1ListExecutionSchedulesForAUser,
  useListCopilotFollowupSchedules,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { okData } from "@/app/api/helpers";
import { isSchedulerUnavailableError } from "@/services/schedules/scheduler-availability";
import { useSchedulerQueryOptions } from "@/services/schedules/useSchedulerQueryOptions";

export type ScheduleItem =
  | { kind: "copilot_turn"; item: CopilotTurnJobInfo }
  | { kind: "graph"; item: GraphExecutionJobInfo };

function nextRunMs(item: { next_run_time?: string | null }): number {
  if (!item.next_run_time) return Number.POSITIVE_INFINITY;
  const t = new Date(item.next_run_time).valueOf();
  return Number.isNaN(t) ? Number.POSITIVE_INFINITY : t;
}

export function useSchedulesPanel() {
  const schedulerQueryOptions = useSchedulerQueryOptions();
  // Followups (copilot_turn) — the scheduled-message side of the
  // feature.  Stored via ``schedule_followup`` MCP tool.
  const copilotQuery = useListCopilotFollowupSchedules({
    query: {
      select: (res) => okData(res) ?? [],
      ...schedulerQueryOptions,
    },
  });
  // Graph schedules — recurring agent runs created via the agent
  // builder.  Same scheduler, different ``kind`` discriminator.  This
  // page now unifies both so users see ALL their pending automated
  // work in one place (briefing pill, this list, single Delete flow).
  const graphQuery = useGetV1ListExecutionSchedulesForAUser({
    query: {
      select: (res) => okData(res) ?? [],
      ...schedulerQueryOptions,
    },
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

  const isLoading = copilotQuery.isLoading || graphQuery.isLoading;
  // A restarting scheduler is its own calm state, never an error or an
  // empty list.
  // Each query's own scheduler 503 is the calm state; any other failure on
  // either query is still a real error, not hidden behind the notice.
  const isUnavailable =
    isSchedulerUnavailableError(copilotQuery.error) ||
    isSchedulerUnavailableError(graphQuery.error);
  const fetchError =
    realError(copilotQuery.error) ?? realError(graphQuery.error);

  function retry() {
    copilotQuery.refetch();
    graphQuery.refetch();
  }

  return {
    // Backwards-compat alias — ``followups`` was the only-copilot
    // collection name before unification.  Existing tests still
    // reference it.
    followups: copilotQuery.data ?? [],
    schedules,
    isLoading,
    isUnavailable,
    retry,
    // The two sources are independent — only hard-fail once both have
    // settled and there is nothing to show; a partial failure keeps the
    // loaded list visible and surfaces a non-blocking warning instead.
    error: !isLoading && schedules.length === 0 ? fetchError : null,
    partialError: !isLoading && schedules.length > 0 ? fetchError : null,
  };
}

function realError<TError>(error: TError | null): TError | null {
  return isSchedulerUnavailableError(error) ? null : error;
}
