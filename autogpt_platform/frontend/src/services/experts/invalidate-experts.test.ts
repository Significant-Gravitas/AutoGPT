import {
  getGetExpertQueryKey,
  getListExpertCredentialsQueryKey,
  getListExpertIdentitiesQueryKey,
  getListExpertSetupItemsQueryKey,
  getListExpertsQueryKey,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { getGetV1ListExecutionSchedulesForAUserQueryKey } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import {
  invalidateExpertGrantQueries,
  invalidateExpertRosterQueries,
} from "./invalidate-experts";

describe("invalidateExpertRosterQueries", () => {
  it("invalidates both the full roster and chat identity projection", async () => {
    const queryClient = new QueryClient();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");

    await invalidateExpertRosterQueries(queryClient);

    expect(invalidate).toHaveBeenCalledWith({
      queryKey: getListExpertsQueryKey(),
    });
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: getListExpertIdentitiesQueryKey(),
    });
  });
});

describe("invalidateExpertGrantQueries", () => {
  it("invalidates every list a grant changes, on both team pages", async () => {
    const queryClient = new QueryClient();
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");

    await invalidateExpertGrantQueries(queryClient, "expert-maria");

    for (const queryKey of [
      getListExpertCredentialsQueryKey("expert-maria"),
      getGetExpertQueryKey("expert-maria"),
      getListExpertSetupItemsQueryKey(),
      getListExpertsQueryKey(),
      getGetV1ListExecutionSchedulesForAUserQueryKey(),
    ]) {
      expect(invalidate).toHaveBeenCalledWith({ queryKey });
    }
  });
});
