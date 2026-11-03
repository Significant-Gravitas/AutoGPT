import { getGetV1ListAllExecutionsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { getGetV1ListAllExecutionsMockHandler200 } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import BackendAPI from "@/lib/autogpt-server-api/client";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { onlineManager } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  deferred,
  earlyAgent,
  execution,
  firstPageAgents,
  lateAgent,
  libraryKey,
  libraryURL,
  page,
  renderActivity,
  settleRequests,
} from "./fixtures";

afterEach(() => {
  onlineManager.setOnline(true);
  vi.restoreAllMocks();
});

describe("Activity review regressions", () => {
  it("receives completion events when the workflow is absent from the library", async () => {
    const events = vi.spyOn(BackendAPI.prototype, "onWebSocketMessage");
    server.use(http.get(libraryURL, () => HttpResponse.json(page([], 1, 0))));
    const running = execution({ status: "RUNNING", ended_at: null });
    renderActivity([running]);
    await screen.findByText("Workflow abcdef01");
    await waitFor(() => expect(events).toHaveBeenCalled());
    const callback = events.mock.calls.find(
      ([method]) => method === "graph_execution_event",
    )?.[1];
    expect(callback).toBeDefined();
    act(() =>
      callback?.({
        ...execution(),
        node_executions: [],
      } as unknown as GraphExecution),
    );
    await screen.findByText(/Completed just now/);
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("keeps a live completion when a same-count execution refresh is stale", async () => {
    const events = vi.spyOn(BackendAPI.prototype, "onWebSocketMessage");
    server.use(
      http.get(libraryURL, () => HttpResponse.json(page([earlyAgent], 1, 1))),
    );
    const running = execution({
      id: "early-execution",
      graph_id: earlyAgent.graph_id,
      status: "RUNNING",
      ended_at: null,
    });
    const other = execution({ id: "other-execution" });
    const { queryClient } = renderActivity([running, other]);
    await screen.findByRole("link", { name: /Early workflow.*running/ });
    const callback = events.mock.calls.find(
      ([method]) => method === "graph_execution_event",
    )?.[1];
    act(() =>
      callback?.({
        ...execution({ id: running.id, graph_id: running.graph_id }),
        node_executions: [],
      } as unknown as GraphExecution),
    );
    await screen.findByRole("link", { name: /Early workflow.*Completed/ });
    server.use(
      getGetV1ListAllExecutionsMockHandler200([
        running,
        { ...other, ended_at: new Date(Date.now() + 10000) },
      ]),
    );
    await queryClient.invalidateQueries({
      queryKey: getGetV1ListAllExecutionsQueryKey(),
    });
    await settleRequests();
    expect(
      screen.queryByRole("link", { name: /Early workflow.*running/ }),
    ).toBeNull();
    expect(
      screen.getByRole("link", { name: /Early workflow.*Completed/ }),
    ).toBeDefined();
  });

  it("continues past repeated records when the page number advances", async () => {
    const requests: number[] = [];
    server.use(
      http.get(libraryURL, ({ request }) => {
        const current = Number(new URL(request.url).searchParams.get("page"));
        requests.push(current);
        return HttpResponse.json(
          page(current < 3 ? firstPageAgents : [lateAgent], current, 201),
        );
      }),
    );
    const { queryClient } = renderActivity();
    await screen.findByRole("link", { name: /Late-page workflow/ });
    await waitFor(() =>
      expect(queryClient.isFetching({ queryKey: libraryKey })).toBe(0),
    );
    expect(requests).toEqual([1, 2, 3]);
  });

  it("recovers an initial library failure when connectivity returns", async () => {
    let fail = true;
    server.use(
      http.get(libraryURL, () =>
        fail
          ? HttpResponse.json({ detail: "Unavailable" }, { status: 503 })
          : HttpResponse.json(page([lateAgent], 1, 1)),
      ),
    );
    renderActivity();
    await screen.findByRole("button", { name: "Try Again" });
    fail = false;
    act(() => onlineManager.setOnline(false));
    act(() => onlineManager.setOnline(true));
    await screen.findByRole("link", { name: /Late-page workflow/ });
  });

  it("loads page two after retrying an unchanged first page", async () => {
    const next = deferred();
    let failRefresh = false;
    const requests: number[] = [];
    server.use(
      http.get(libraryURL, async ({ request }) => {
        const current = Number(new URL(request.url).searchParams.get("page"));
        requests.push(current);
        if (current === 1)
          return failRefresh
            ? HttpResponse.json({ detail: "Unavailable" }, { status: 503 })
            : HttpResponse.json(page(firstPageAgents));
        await next.promise;
        return HttpResponse.json(page([lateAgent], 2));
      }),
    );
    const { queryClient } = renderActivity();
    await waitFor(() => expect(requests).toEqual([1, 2]));
    failRefresh = true;
    await queryClient.invalidateQueries({ queryKey: libraryKey });
    next.resolve();
    await screen.findByRole("button", { name: "Try Again" });
    const attempts = requests.length;
    failRefresh = false;
    fireEvent.click(screen.getByRole("button", { name: "Try Again" }));
    await screen.findByRole("link", { name: /Late-page workflow/ });
    await waitFor(() =>
      expect(queryClient.isFetching({ queryKey: libraryKey })).toBe(0),
    );
    expect(requests.slice(attempts)).toEqual([1, 2]);
  });
});
