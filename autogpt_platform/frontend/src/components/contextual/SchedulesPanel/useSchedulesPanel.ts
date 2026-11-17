import {
  useGetV1ListExecutionSchedulesForAUser,
  useListCopilotFollowupSchedules,
  useListPausedScheduledChats,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { okData } from "@/app/api/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

export type ScheduleItem =
  | { kind: "copilot_turn"; item: CopilotTurnJobInfo }
  | { kind: "graph"; item: GraphExecutionJobInfo };

// The scheduler restarts on every deploy and answers 503 while it is out: the
// schedules are unknown for a moment, neither missing nor broken.
function isSchedulerUnavailable(error: unknown) {
  return error instanceof ApiError && error.status === 503;
}

function nextRunMs(item: { next_run_time?: string | null }): number {
  if (!item.next_run_time) return Number.POSITIVE_INFINITY;
  const t = new Date(item.next_run_time).valueOf();
  return Number.isNaN(t) ? Number.POSITIVE_INFINITY : t;
}

export function useSchedulesPanel() {
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
  const isLoading =
    copilotQuery.isLoading || graphQuery.isLoading || pausedQuery.isLoading;
  const hasContent = schedules.length > 0 || paused.length > 0;
  const schedulerErrors = [copilotQuery.error, graphQuery.error];
  const outage = schedulerErrors.find(isSchedulerUnavailable) ?? null;
  const fetchError =
    schedulerErrors.find((error) => error && !isSchedulerUnavailable(error)) ??
    pausedQuery.error;
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
    // Nothing to show, and only because the scheduler is out: a calm
    // notice, never an error or an empty list.
    unavailable: settled && !hasContent && !fetchError && outage !== null,
    retryUnavailable,
    retrying: copilotQuery.isFetching || graphQuery.isFetching,
    // The sources are independent — only hard-fail once all have settled
    // and there is nothing to show; a partial failure, an outage included,
    // keeps the loaded lists visible and surfaces a non-blocking warning.
    error: settled && !hasContent ? fetchError : null,
    partialError: settled && hasContent ? (fetchError ?? outage) : null,
  };
}
