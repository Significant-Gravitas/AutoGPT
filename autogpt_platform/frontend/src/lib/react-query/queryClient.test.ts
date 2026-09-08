import { QueryClient, QueryObserver } from "@tanstack/react-query";
import { describe, expect, test, vi } from "vitest";
import { resetQueryClientForIdentityChange } from "./queryClient";

describe("resetQueryClientForIdentityChange", () => {
  test("hides active query data without refetching after logout", async () => {
    const queryClient = new QueryClient();
    const queryKey = ["billing-portal"];
    const queryFn = vi.fn(async () => "new-session");
    queryClient.setQueryData(queryKey, "old-session");
    const observer = new QueryObserver(queryClient, {
      enabled: false,
      queryFn,
      queryKey,
    });
    const unsubscribe = observer.subscribe(() => {});

    await resetQueryClientForIdentityChange(null, queryClient);

    expect(observer.getCurrentResult().data).toBeUndefined();
    expect(queryFn).not.toHaveBeenCalled();
    unsubscribe();
  });

  test("resets and refetches active queries for a new identity", async () => {
    const queryClient = new QueryClient();
    const queryKey = ["billing-portal"];
    const queryFn = vi.fn(async () => "new-session");
    queryClient.setQueryData(queryKey, "old-session");
    const observer = new QueryObserver(queryClient, {
      queryFn,
      queryKey,
      staleTime: Infinity,
    });
    const unsubscribe = observer.subscribe(() => {});

    await resetQueryClientForIdentityChange("user-b", queryClient);

    expect(observer.getCurrentResult().data).toBe("new-session");
    expect(queryFn).toHaveBeenCalledTimes(1);
    unsubscribe();
  });
});
