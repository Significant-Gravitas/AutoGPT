import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { describe, expect, test } from "vitest";
import { server } from "@/mocks/mock-server";
import { useCopilotPendingReviews } from "../useCopilotPendingReviews";

function countRequests(status: string) {
  const calls = { execution: 0, reviews: 0 };
  server.use(
    http.get("*/api/graphs/graph-1/executions/exec-1", () => {
      calls.execution++;
      return HttpResponse.json({ id: "exec-1", graph_id: "graph-1", status });
    }),
    http.get("*/api/review/execution/exec-1", () => {
      calls.reviews++;
      return HttpResponse.json([]);
    }),
  );
  return calls;
}

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe("useCopilotPendingReviews", () => {
  test("stops polling once the run has finished", async () => {
    const calls = countRequests("COMPLETED");
    renderHook(
      () =>
        useCopilotPendingReviews({ graphExecId: "exec-1", graphId: "graph-1" }),
      { wrapper },
    );
    await waitFor(() => expect(calls.reviews).toBeGreaterThan(0));
    const settled = { ...calls };

    await sleep(4500);

    expect(calls).toEqual(settled);
  }, 10_000);

  test("polls a run that has not paused yet slowly", async () => {
    const calls = countRequests("RUNNING");
    renderHook(
      () =>
        useCopilotPendingReviews({ graphExecId: "exec-1", graphId: "graph-1" }),
      { wrapper },
    );
    await waitFor(() => expect(calls.execution).toBeGreaterThan(0));
    const settled = calls.execution;

    await sleep(4000);

    expect(calls.execution).toBe(settled);
  }, 10_000);

  test("still polls the reviews when the run's status cannot be read", async () => {
    const calls = { reviews: 0 };
    server.use(
      http.get("*/api/graphs/graph-1/executions/exec-1", () =>
        HttpResponse.json({ detail: "down" }, { status: 500 }),
      ),
      http.get("*/api/review/execution/exec-1", () => {
        calls.reviews++;
        return HttpResponse.json([]);
      }),
    );
    renderHook(
      () =>
        useCopilotPendingReviews({ graphExecId: "exec-1", graphId: "graph-1" }),
      { wrapper },
    );

    await waitFor(() => expect(calls.reviews).toBeGreaterThan(1), {
      timeout: 8000,
    });
  }, 10_000);

  test("a chat's queue is polled from the chat, never from a run", async () => {
    const calls = { chat: 0, execution: 0, runReviews: 0 };
    server.use(
      http.get("*/api/review/session/chat-1", () => {
        calls.chat++;
        return HttpResponse.json([{ node_exec_id: "chat-review" }]);
      }),
      http.get("*/api/graphs/*", () => {
        calls.execution++;
        return HttpResponse.json({});
      }),
      http.get("*/api/review/execution/*", () => {
        calls.runReviews++;
        return HttpResponse.json([]);
      }),
    );
    const { result } = renderHook(
      () => useCopilotPendingReviews({ chatSessionId: "chat-1" }),
      { wrapper },
    );

    await waitFor(() => expect(calls.chat).toBeGreaterThan(1), {
      timeout: 5000,
    });
    expect(result.current.pendingReviews).toEqual([
      { node_exec_id: "chat-review" },
    ]);
    expect(calls.execution).toBe(0);
    expect(calls.runReviews).toBe(0);
  }, 10_000);

  test("keeps polling reviews while the run is paused for one", async () => {
    const calls = countRequests("REVIEW");
    renderHook(
      () =>
        useCopilotPendingReviews({ graphExecId: "exec-1", graphId: "graph-1" }),
      { wrapper },
    );
    await waitFor(() => expect(calls.reviews).toBeGreaterThan(0));
    const settled = calls.reviews;

    await sleep(4500);

    expect(calls.reviews).toBeGreaterThan(settled);
  }, 10_000);

  test.each([
    { rows: [], polls: false },
    { rows: [{ node_exec_id: "copilot-node-gate-x:1" }], polls: true },
  ])(
    "a chat with nothing held on screen polls its cards only once it has one ($rows.length)",
    async ({ rows, polls }) => {
      let requests = 0;
      server.use(
        http.get("*/api/review/session/s1", () => {
          requests++;
          return HttpResponse.json(rows);
        }),
      );
      renderHook(
        () =>
          useCopilotPendingReviews({
            chatSessionId: "s1",
            pollWhileEmpty: false,
          }),
        { wrapper },
      );
      await waitFor(() => expect(requests).toBeGreaterThan(0));

      await sleep(4500);

      expect(requests > 1).toBe(polls);
    },
    15_000,
  );

  test("a new held call on screen fetches the chat's cards again", async () => {
    let requests = 0;
    server.use(
      http.get("*/api/review/session/s1", () => {
        requests++;
        return HttpResponse.json([]);
      }),
    );
    const { rerender } = renderHook(
      ({ key }) =>
        useCopilotPendingReviews({
          chatSessionId: "s1",
          pollWhileEmpty: false,
          refetchKey: key,
        }),
      { wrapper, initialProps: { key: 0 } },
    );
    await waitFor(() => expect(requests).toBeGreaterThan(0));
    await sleep(300);
    const settled = requests;

    rerender({ key: 1 });

    await waitFor(() => expect(requests).toBeGreaterThan(settled));
  }, 10_000);
});
