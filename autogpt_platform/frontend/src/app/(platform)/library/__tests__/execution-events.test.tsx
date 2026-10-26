import {
  afterEach,
  beforeEach,
  describe,
  expect,
  test,
  vi,
  type MockInstance,
} from "vitest";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import BackendAPI from "@/lib/autogpt-server-api/client";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import LibraryPage from "../page";

type ExecutionHandler = (execution: GraphExecution) => void;

const executionHandlers = new Set<ExecutionHandler>();

function makeAgent(index: number): LibraryAgent {
  const base = getGetV2ListLibraryAgentsResponseMock().agents[0];
  return {
    ...base,
    id: `agent-${index}`,
    graph_id: `graph-${index}`,
    name: `Workflow ${index}`,
  };
}

const AGENTS = [makeAgent(1), makeAgent(2), makeAgent(3)];

function setupHandlers() {
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...getGetV2ListLibraryAgentsResponseMock(),
      agents: AGENTS,
      pagination: {
        total_items: AGENTS.length,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV1ListAllExecutionsMockHandler([]),
  );
}

function emitExecutionEvent(graphID: string, runID: string) {
  const execution = {
    id: runID,
    graph_id: graphID,
    status: "RUNNING",
  } as unknown as GraphExecution;
  act(() => {
    executionHandlers.forEach((handler) => handler(execution));
  });
}

function countExecutionListRequests() {
  const counter = { count: 0 };
  function record({ request }: { request: Request }) {
    if (
      request.method === "GET" &&
      new URL(request.url).pathname.endsWith("/api/executions")
    ) {
      counter.count += 1;
    }
  }
  server.events.on("request:start", record);
  return {
    counter,
    stop: () => server.events.removeListener("request:start", record),
  };
}

let subscribe: MockInstance<BackendAPI["subscribeToGraphExecutions"]>;

beforeEach(() => {
  executionHandlers.clear();
  subscribe = vi
    .spyOn(BackendAPI.prototype, "subscribeToGraphExecutions")
    .mockResolvedValue(undefined);
  vi.spyOn(BackendAPI.prototype, "connectWebSocket").mockResolvedValue(
    undefined,
  );
  vi.spyOn(BackendAPI.prototype, "onWebSocketConnect").mockImplementation(
    (handler: () => void) => {
      handler();
      return () => undefined;
    },
  );
  vi.spyOn(BackendAPI.prototype, "onWebSocketMessage").mockImplementation(((
    method: string,
    handler: ExecutionHandler,
  ) => {
    if (method !== "graph_execution_event") return () => undefined;
    executionHandlers.add(handler);
    return () => executionHandlers.delete(handler);
  }) as unknown as BackendAPI["onWebSocketMessage"]);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("LibraryPage execution events", () => {
  test("re-rendering the page with the same data does not re-subscribe", async () => {
    setupHandlers();
    const { rerender } = render(<LibraryPage />);
    await screen.findByText("Workflow 3");
    await waitFor(() => expect(subscribe).toHaveBeenCalledTimes(3));

    rerender(<LibraryPage />);
    rerender(<LibraryPage />);
    rerender(<LibraryPage />);
    await act(async () => undefined);

    expect(subscribe.mock.calls.map(([graphID]) => graphID).sort()).toEqual([
      "graph-1",
      "graph-2",
      "graph-3",
    ]);
  });

  test("a burst of execution events refetches the run list at most twice", async () => {
    setupHandlers();
    render(<LibraryPage />);
    await screen.findByText("Workflow 3");
    await waitFor(() => expect(executionHandlers.size).toBeGreaterThan(0));

    const requests = countExecutionListRequests();
    try {
      for (let index = 0; index < 10; index += 1) {
        emitExecutionEvent(`graph-${(index % 3) + 1}`, `run-${index}`);
      }
      await waitFor(() => expect(requests.counter.count).toBe(1));
      await waitFor(() => expect(requests.counter.count).toBe(2), {
        timeout: 4000,
      });
    } finally {
      requests.stop();
    }
  });
});
