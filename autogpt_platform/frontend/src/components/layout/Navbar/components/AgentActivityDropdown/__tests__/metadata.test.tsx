import { getGetV1ListAllExecutionsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import BackendAPI from "@/lib/autogpt-server-api/client";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { server } from "@/mocks/mock-server";
import { screen } from "@/tests/integrations/test-utils";
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

afterEach(() => vi.restoreAllMocks());

describe("Activity metadata", () => {
  it("shows deleted workflows even when the entire library is empty", async () => {
    const requests = vi.fn(() => HttpResponse.json(page([], 1, 0)));
    server.use(http.get(libraryURL, requests));
    renderActivity();
    await screen.findByText("Workflow abcdef01");
    expect(screen.queryByRole("link")).toBeNull();
    await settleRequests();
    expect(requests).toHaveBeenCalledTimes(1);
  });

  it("keeps unavailable workflow activity without inventing a library link", async () => {
    server.use(
      http.get(libraryURL, () => HttpResponse.json(page([earlyAgent], 1, 1))),
    );
    renderActivity();
    await screen.findByText("Workflow abcdef01");
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("refreshes metadata with unchanged IDs and removes links for deleted workflows", async () => {
    let agents = [lateAgent];
    server.use(
      http.get(libraryURL, () =>
        HttpResponse.json(page(agents, 1, agents.length)),
      ),
    );
    const { queryClient } = renderActivity();
    await screen.findByRole("link", { name: /Late-page workflow/ });
    agents = [{ ...lateAgent, name: "<script>Renamed workflow</script>" }];
    await queryClient.invalidateQueries({ queryKey: libraryKey });
    await screen.findByRole("link", {
      name: /<script>Renamed workflow<\/script>/,
    });
    expect(document.querySelector("script")).toBeNull();
    agents = [];
    await queryClient.invalidateQueries({ queryKey: libraryKey });
    await screen.findByText("Workflow abcdef01");
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("enriches late metadata without reverting a live execution completion", async () => {
    const next = deferred();
    const events = vi.spyOn(BackendAPI.prototype, "onWebSocketMessage");
    server.use(
      http.get(libraryURL, async ({ request }) => {
        const current = Number(new URL(request.url).searchParams.get("page"));
        if (current === 1) return HttpResponse.json(page(firstPageAgents));
        await next.promise;
        return HttpResponse.json(page([lateAgent], 2));
      }),
    );
    const running = execution({
      id: "early-execution",
      graph_id: earlyAgent.graph_id,
      status: "RUNNING",
      ended_at: null,
    });
    const { queryClient } = renderActivity([running, execution()]);
    await screen.findByRole("link", { name: /Early workflow.*running/ });
    const callback = events.mock.calls.find(
      ([method]) => method === "graph_execution_event",
    )?.[1];
    expect(callback).toBeDefined();
    callback?.({
      ...running,
      ...execution({ id: "early-execution", graph_id: earlyAgent.graph_id }),
      node_executions: [],
    } as unknown as GraphExecution);
    await screen.findByRole("link", { name: /Early workflow.*Completed/ });
    next.resolve();
    await screen.findByRole("link", { name: /Late-page workflow/ });
    expect(
      screen.queryByRole("link", { name: /Early workflow.*running/ }),
    ).toBeNull();
    expect(
      screen.getByRole("link", { name: /Early workflow.*Completed/ }),
    ).toBeDefined();
    expect(
      queryClient.getQueryState(getGetV1ListAllExecutionsQueryKey())?.status,
    ).toBe("success");
  });
});
