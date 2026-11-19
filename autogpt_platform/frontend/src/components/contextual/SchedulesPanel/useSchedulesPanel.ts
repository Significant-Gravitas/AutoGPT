import {
  useGetV1ListExecutionSchedulesForAUser,
  useListCopilotFollowupSchedules,
  useListPausedScheduledChats,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { okData } from "@/app/api/helpers";
import { isSchedulerUnavailableError } from "@/services/schedules/scheduler-availability";
import { useSchedulerOutage } from "@/services/schedules/useSchedulerOutage";
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
  // A 503 means the scheduler is restarting: recheck after its Retry-After.
  const schedulerQueryOptions = useSchedulerQueryOptions();
  // Followups (copilot_turn) — the scheduled-message side of the
  // feature.  Stored via ``schedule_followup`` MCP tool.
  const copilotQuery = useListCopilotFollowupSchedules({
    query: { select: (res) => okData(res) ?? [], ...schedulerQueryOptions },
  });
  // Graph schedules — recurring agent runs created via the agent
  // builder.  Same scheduler, different ``kind`` discriminator.  This
  // page now unifies both so users see ALL their pending automated
  // work in one place (briefing pill, this list, single Delete flow).
  const graphQuery = useGetV1ListExecutionSchedulesForAUser({
    query: { select: (res) => okData(res) ?? [], ...schedulerQueryOptions },
  });
  // Scheduled chats paused after failing the same way: their jobs drop out
  // of the followups list, so they are listed apart, with Resume.
  const pausedQuery = useListPausedScheduledChats({
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

  const paused = pausedQuery.data ?? [];
  // A list rechecking after an outage stays out, not loading, until the
  // scheduler answers.
  const copilotOut = useSchedulerOutage(copilotQuery);
  const graphOut = useSchedulerOutage(graphQuery);
  const outage = copilotOut || graphOut;
  const isLoading =
    (copilotQuery.isLoading && !copilotOut) ||
    (graphQuery.isLoading && !graphOut) ||
    pausedQuery.isLoading;
  const hasContent = schedules.length > 0 || paused.length > 0;
  const fetchError =
    [copilotQuery.error, graphQuery.error].find(
      (error) => error && !isSchedulerUnavailableError(error),
    ) ?? pausedQuery.error;
  const settled = !isLoading;

  function retryUnavailable() {
    copilotQuery.refetch();
    graphQuery.refetch();
  }

  return {
    // Backwards-compat alias — ``followups`` was the only-copilot
    // collection name before unification.  Existing tests still
    // reference it.
    followups: copilotQuery.data ?? [],
    schedules,
    paused,
    isLoading,
    // The scheduler is out: a calm notice, never an error or an empty list,
    // above whatever did load.
    unavailable: settled && !fetchError && outage,
    retryUnavailable,
    retrying: copilotQuery.isFetching || graphQuery.isFetching,
    // The sources are independent — only hard-fail once all have settled
    // and there is nothing to show; a partial failure keeps the loaded lists
    // visible and surfaces a non-blocking warning.
    error: settled && !hasContent ? fetchError : null,
    partialError: settled && hasContent ? fetchError : null,
  };
}
