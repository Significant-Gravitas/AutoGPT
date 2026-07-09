import {
  getGetV1ListExecutionSchedulesForAGraphQueryKey,
  getGetV1ListExecutionSchedulesForAUserQueryKey,
  getListCopilotFollowupSchedulesQueryKey,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { QueryClient } from "@tanstack/react-query";
import { describe, expect, test, vi } from "vitest";
import { invalidateAllScheduleQueries } from "./invalidate-schedules";

describe("invalidateAllScheduleQueries", () => {
  test("invalidates user-wide schedules + copilot followups + library agents + per-graph when graphId provided", () => {
    const queryClient = new QueryClient();
    const spy = vi.spyOn(queryClient, "invalidateQueries");

    invalidateAllScheduleQueries(queryClient, "graph-abc");

    expect(spy).toHaveBeenCalledTimes(4);
    expect(spy).toHaveBeenCalledWith({
      queryKey: getGetV1ListExecutionSchedulesForAUserQueryKey(),
    });
    expect(spy).toHaveBeenCalledWith({
      queryKey: getListCopilotFollowupSchedulesQueryKey(),
    });
    expect(spy).toHaveBeenCalledWith({
      queryKey: ["/api/library/agents"],
    });
    expect(spy).toHaveBeenCalledWith({
      queryKey: getGetV1ListExecutionSchedulesForAGraphQueryKey("graph-abc"),
    });
  });

  test("skips per-graph invalidation when graphId is omitted (followup-only mutations don't know a graph)", () => {
    const queryClient = new QueryClient();
    const spy = vi.spyOn(queryClient, "invalidateQueries");

    invalidateAllScheduleQueries(queryClient);

    expect(spy).toHaveBeenCalledTimes(3);
    expect(spy).toHaveBeenCalledWith({
      queryKey: getGetV1ListExecutionSchedulesForAUserQueryKey(),
    });
    expect(spy).toHaveBeenCalledWith({
      queryKey: getListCopilotFollowupSchedulesQueryKey(),
    });
    expect(spy).toHaveBeenCalledWith({
      queryKey: ["/api/library/agents"],
    });
  });
});
