import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import type { ReactNode } from "react";
import { expect, test, vi } from "vitest";
import {
  getGetV2GetPendingReviewsForExecutionQueryKey,
  getGetV2GetPendingReviewsQueryKey,
} from "@/app/api/__generated__/endpoints/executions/executions";
import { getPostV2ProcessReviewActionMockHandler200 } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { server } from "@/mocks/mock-server";
import { useProcessReviews } from "./useProcessReviews";

function setup() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const spy = vi.spyOn(queryClient, "invalidateQueries");

  function wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }

  const { result } = renderHook(() => useProcessReviews(), { wrapper });
  return {
    result,
    invalidatedKeys: () => spy.mock.calls.map((c) => c[0]?.queryKey),
  };
}

const item = {
  node_exec_id: "ne-1",
  approved: true,
  auto_approve_future: false,
};

// Dual-key invalidation is what makes the home count and the thread's list
// reconcile after either surface acts on a review; without it one of them
// keeps serving a review that has already been processed.
test("invalidates both the user-wide and the per-execution review queries", async () => {
  server.use(
    getPostV2ProcessReviewActionMockHandler200({
      approved_count: 1,
      rejected_count: 0,
      failed_count: 0,
    }),
  );
  const { result, invalidatedKeys } = setup();

  await result.current.processReviews([item], ["run-1", "run-1"]);

  const keys = invalidatedKeys();
  expect(keys).toContainEqual(getGetV2GetPendingReviewsQueryKey());
  const perExecution = getGetV2GetPendingReviewsForExecutionQueryKey("run-1");
  expect(keys).toContainEqual(perExecution);
  // Deduped: the same execution id twice must not fan out twice.
  expect(
    keys.filter((key) => JSON.stringify(key) === JSON.stringify(perExecution)),
  ).toHaveLength(1);
});

test("still invalidates when the mutation rejects", async () => {
  server.use(
    http.post("/api/proxy/api/review/action", () => HttpResponse.error()),
  );
  const { result, invalidatedKeys } = setup();

  await expect(
    result.current.processReviews([item], ["run-1"]),
  ).rejects.toBeDefined();

  expect(invalidatedKeys()).toContainEqual(getGetV2GetPendingReviewsQueryKey());
});
