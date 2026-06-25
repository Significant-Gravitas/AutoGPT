import {
  getGetV1ListExecutionSchedulesForAGraphQueryKey,
  getGetV1ListExecutionSchedulesForAUserQueryKey,
  getListCopilotFollowupSchedulesQueryKey,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { QueryClient } from "@tanstack/react-query";

// Schedule mutations (create / edit / delete) need to invalidate every
// list query that might cache the affected row.  There are four:
//   - user-wide /api/v1/schedules — unified `/library/followups` page
//     AND the "Autopilot library" briefing pill count.
//   - per-graph /api/v1/graphs/{id}/schedules — agent detail page
//     sidebar + selected-schedule view.
//   - copilot followups list — same scheduler primitive, separate
//     endpoint; included so a followup-side mutation doesn't leave
//     graph schedule caches stale and vice versa.
//   - library agents list (/api/library/agents) — backend computes
//     `is_scheduled` on the row, so creating/deleting a schedule flips
//     that flag and the `/library` fleet-summary "Scheduled" count
//     would otherwise stay stale until manual reload.
//
// Forgetting any one of these leaves a stale row visible on another
// page until manual reload, which we saw with the unified Scheduled
// page after PR #13202.  Always invalidate ALL of them.
export function invalidateAllScheduleQueries(
  queryClient: QueryClient,
  graphId?: string,
) {
  queryClient.invalidateQueries({
    queryKey: getGetV1ListExecutionSchedulesForAUserQueryKey(),
  });
  queryClient.invalidateQueries({
    queryKey: getListCopilotFollowupSchedulesQueryKey(),
  });
  // Partial-key match: hits every variant of the library agents query
  // (search, filter, pagination, plus the infinite-query form used by
  // `/library`'s agent list).  Required so `agent.is_scheduled` —
  // computed server-side and fed into the fleet-summary "Scheduled"
  // count — refreshes after a schedule mutation.
  queryClient.invalidateQueries({ queryKey: ["/api/library/agents"] });
  if (graphId) {
    queryClient.invalidateQueries({
      queryKey: getGetV1ListExecutionSchedulesForAGraphQueryKey(graphId),
    });
  }
}
