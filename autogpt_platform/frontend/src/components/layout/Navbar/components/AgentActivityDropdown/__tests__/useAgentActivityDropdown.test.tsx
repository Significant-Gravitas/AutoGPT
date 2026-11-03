import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  getGetV1ListAllExecutionsMockHandler,
  getGetV1ListAllExecutionsResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetV2ListLibraryAgentsResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { server } from "@/mocks/mock-server";
import { useAgentActivityDropdown } from "../useAgentActivityDropdown";

const executionHandlers = new Set<(execution: GraphExecution) => void>();

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => ({
    onWebSocketConnect: () => () => {},
    onWebSocketDisconnect: () => () => {},
    onWebSocketMessage: (
      _method: string,
      handler: (execution: GraphExecution) => void,
    ) => {
      executionHandlers.add(handler);
      return () => executionHandlers.delete(handler);
    },
    subscribeToGraphExecutions: () => Promise.resolve(),
    connectWebSocket: () => Promise.resolve(),
  }),
}));

let releaseHeldPage: (() => void) | undefined;

beforeEach(() => {
  executionHandlers.clear();
});

// Settle the held page even when a test fails before releasing it.
afterEach(() => {
  releaseHeldPage?.();
  releaseHeldPage = undefined;
});

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }
  return Wrapper;
}

const agentTemplate = getGetV2ListLibraryAgentsResponseMock().agents[0];

// 150 workflows over two pages of 100. Page 2 is held until `releasePage2`
// is called, so the test controls when the rest of the library arrives.
function serveLibrary() {
  let releasePage2 = () => {};
  const page2Released = new Promise<void>((resolve) => {
    releasePage2 = resolve;
  });
  releaseHeldPage = releasePage2;
  server.use(
    http.get("*/api/library/agents", async ({ request }) => {
      const params = new URL(request.url).searchParams;
      const page = Number(params.get("page") ?? 1);
      const pageSize = Number(params.get("page_size") ?? 15);
      if (page > 1) await page2Released;
      const start = (page - 1) * pageSize;
      const count = Math.max(0, Math.min(pageSize, 150 - start));
      return HttpResponse.json({
        agents: Array.from({ length: count }, (_, i) => ({
          ...agentTemplate,
          id: `la-${start + i}`,
          graph_id: `g-${start + i}`,
          name: `Workflow number ${start + i}`,
        })),
        pagination: {
          total_items: 150,
          total_pages: Math.ceil(150 / pageSize),
          current_page: page,
          page_size: pageSize,
        },
      });
    }),
  );
  return () => releasePage2();
}

function runningExecution(id: string, graphID: string) {
  return {
    ...getGetV1ListAllExecutionsResponseMock()[0],
    id,
    graph_id: graphID,
    status: AgentExecutionStatus.RUNNING,
    started_at: new Date(),
    ended_at: null,
  };
}

describe("useAgentActivityDropdown", () => {
  test("names a run of a workflow past the first library page", async () => {
    const releasePage2 = serveLibrary();
    server.use(
      getGetV1ListAllExecutionsMockHandler([
        runningExecution("run-1", "g-120"),
      ]),
    );

    const { result } = renderHook(() => useAgentActivityDropdown(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current.activeExecutions[0]?.agent_name).toBe(
        "Unknown Workflow",
      ),
    );
    releasePage2();
    await waitFor(() =>
      expect(result.current.activeExecutions[0]?.agent_name).toBe(
        "Workflow number 120",
      ),
    );
  });

  test("keeps a live update that arrived before the next library page", async () => {
    const releasePage2 = serveLibrary();
    const pageOneRun = runningExecution("run-1", "g-5");
    server.use(
      getGetV1ListAllExecutionsMockHandler([
        pageOneRun,
        runningExecution("run-2", "g-120"),
      ]),
    );

    const { result } = renderHook(() => useAgentActivityDropdown(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current.activeExecutions).toHaveLength(2),
    );
    await waitFor(() => expect(executionHandlers.size).toBeGreaterThan(0));
    act(() => {
      executionHandlers.forEach((handler) =>
        handler({
          ...pageOneRun,
          status: AgentExecutionStatus.COMPLETED,
          ended_at: new Date(),
        } as unknown as GraphExecution),
      );
    });
    await waitFor(() =>
      expect(result.current.recentCompletions).toHaveLength(1),
    );

    releasePage2();
    await waitFor(() =>
      expect(result.current.activeExecutions[0]?.agent_name).toBe(
        "Workflow number 120",
      ),
    );
    expect(result.current.activeExecutions).toHaveLength(1);
    expect(result.current.recentCompletions[0]?.id).toBe("run-1");
  });
});
