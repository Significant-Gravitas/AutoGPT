import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { delay, http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { describe, expect, test, vi } from "vitest";

import {
  getGetV1ListAllExecutionsMockHandler,
  getGetV1ListAllExecutionsResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetV2ListLibraryAgentsResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { server } from "@/mocks/mock-server";
import { useAgentActivityDropdown } from "../useAgentActivityDropdown";

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => ({
    onWebSocketConnect: () => () => {},
    onWebSocketDisconnect: () => () => {},
    onWebSocketMessage: () => () => {},
    subscribeToGraphExecutions: () => Promise.resolve(),
    connectWebSocket: () => Promise.resolve(),
  }),
}));

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

// 150 workflows over two pages of 100; page 2 arrives well after the
// executions list, as it does in the browser.
function serveLibrary() {
  server.use(
    http.get("*/api/library/agents", async ({ request }) => {
      const params = new URL(request.url).searchParams;
      const page = Number(params.get("page") ?? 1);
      const pageSize = Number(params.get("page_size") ?? 15);
      if (page > 1) await delay(100);
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
}

describe("useAgentActivityDropdown", () => {
  test("names a run of a workflow past the first library page", async () => {
    serveLibrary();
    const execution = {
      ...getGetV1ListAllExecutionsResponseMock()[0],
      id: "run-1",
      graph_id: "g-120",
      status: AgentExecutionStatus.RUNNING,
    };
    server.use(getGetV1ListAllExecutionsMockHandler([execution]));

    const { result } = renderHook(() => useAgentActivityDropdown(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() =>
      expect(result.current.activeExecutions[0]?.agent_name).toBe(
        "Unknown Workflow",
      ),
    );
    await waitFor(() =>
      expect(result.current.activeExecutions[0]?.agent_name).toBe(
        "Workflow number 120",
      ),
    );
  });
});
