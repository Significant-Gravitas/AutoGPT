import {
  getListExpertIdentitiesQueryKey,
  getListExpertsQueryKey,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { invalidateExpertRosterQueries } from "./invalidate-experts";

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
