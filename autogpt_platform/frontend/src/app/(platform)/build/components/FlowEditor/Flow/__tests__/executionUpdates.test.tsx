import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useEdgeStore } from "../../../../stores/edgeStore";
import { useGraphStore } from "../../../../stores/graphStore";
import { useNodeStore } from "../../../../stores/nodeStore";
import { useFlow } from "../useFlow";
import { useFlowRealtime } from "../useFlowRealtime";

// Stands in for the BackendAPI websocket: the test decides when the socket
// connects, when the server acknowledges the subscription, and what it sends.
const socket = vi.hoisted(() => {
  type Handler = (data: unknown) => void;
  const messageHandlers = new Map<string, Set<Handler>>();
  const connectHandlers = new Set<() => void>();
  const disconnectHandlers = new Set<() => void>();
  const state = { open: false, subscribes: [] as string[] };
  const api = {
    onWebSocketMessage(method: string, handler: Handler) {
      if (!messageHandlers.has(method)) messageHandlers.set(method, new Set());
      messageHandlers.get(method)!.add(handler);
      return () => messageHandlers.get(method)!.delete(handler);
    },
    onWebSocketConnect(handler: () => void) {
      connectHandlers.add(handler);
      if (state.open) handler();
      return () => connectHandlers.delete(handler);
    },
    onWebSocketDisconnect(handler: () => void) {
      disconnectHandlers.add(handler);
      return () => disconnectHandlers.delete(handler);
    },
    subscribeToGraphExecution(executionID: string) {
      state.subscribes.push(executionID);
      return Promise.resolve();
    },
  };
  return {
    api,
    state,
    connect() {
      state.open = true;
      connectHandlers.forEach((handler) => handler());
    },
    disconnect() {
      state.open = false;
      disconnectHandlers.forEach((handler) => handler());
    },
    send(method: string, data?: unknown) {
      messageHandlers.get(method)?.forEach((handler) => handler(data));
    },
    reset() {
      state.open = false;
      state.subscribes = [];
      messageHandlers.clear();
      connectHandlers.clear();
      disconnectHandlers.clear();
    },
  };
});

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => socket.api,
}));

vi.mock("@xyflow/react", async () => {
  const actual = await vi.importActual("@xyflow/react");
  return {
    ...actual,
    useReactFlow: () => ({
      screenToFlowPosition: (position: unknown) => position,
      fitView: vi.fn(),
    }),
  };
});

vi.mock("../../../../hooks/useIsReadOnlyGraph", () => ({
  useIsReadOnlyGraph: () => ({ isReadOnly: false }),
}));

const GRAPH_ID = "graph-1";
const EXECUTION_ID = "execution-1";
const BLOCK_ID = "block-1";
const NODE_IDS = ["node-1", "node-2", "node-3"];
const POLL_INTERVAL_MS = 1_000;
const RECONCILE_INTERVAL_MS = 10_000;

type NodeStatus = "QUEUED" | "RUNNING" | "COMPLETED";

function nodeExecution(nodeID: string, status: NodeStatus) {
  return {
    user_id: "user-1",
    graph_id: GRAPH_ID,
    graph_version: 1,
    graph_exec_id: EXECUTION_ID,
    node_exec_id: `exec-${nodeID}`,
    node_id: nodeID,
    block_id: BLOCK_ID,
    status,
    input_data: { input: nodeID },
    output_data: status === "COMPLETED" ? { output: [nodeID] } : {},
    add_time: "2026-01-05T09:00:00.000Z",
    queue_time: "2026-01-05T09:00:00.000Z",
    start_time: status === "QUEUED" ? null : "2026-01-05T09:00:01.000Z",
    end_time: status === "COMPLETED" ? "2026-01-05T09:00:02.000Z" : null,
  };
}

// What the backend returns for the run; tests change it as the run moves on.
let run: {
  status: "RUNNING" | "COMPLETED";
  nodes: Partial<Record<string, NodeStatus>>;
  earlierExecutions?: ReturnType<typeof nodeExecution>[];
};
let executionRequests: number;
// While set, execution responses are held back after the backend has read
// the run, so they arrive stale.
let heldResponses: Promise<void> | null;

function graphResponse() {
  return {
    id: GRAPH_ID,
    version: 1,
    name: "Test workflow",
    description: "",
    is_active: true,
    nodes: NODE_IDS.map((id, index) => ({
      id,
      block_id: BLOCK_ID,
      input_default: {},
      metadata: { position: { x: index * 400, y: 0 } },
      input_links: [],
      output_links: [],
    })),
    links: [
      {
        id: "link-1",
        source_id: "node-1",
        sink_id: "node-2",
        source_name: "output",
        sink_name: "input",
        is_static: false,
      },
    ],
    input_schema: {},
    output_schema: {},
    credentials_input_schema: {},
  };
}

function executionResponse() {
  return {
    id: EXECUTION_ID,
    graph_id: GRAPH_ID,
    graph_version: 1,
    user_id: "user-1",
    status: run.status,
    started_at: "2026-01-05T09:00:00.000Z",
    ended_at: run.status === "COMPLETED" ? "2026-01-05T09:00:03.000Z" : null,
    inputs: {},
    outputs: {},
    node_executions: [
      ...(run.earlierExecutions ?? []),
      ...Object.entries(run.nodes).map(([nodeID, status]) =>
        nodeExecution(nodeID, status!),
      ),
    ],
  };
}

function renderBuilder() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <NuqsTestingAdapter
          searchParams={`?flowID=${GRAPH_ID}&flowExecutionID=${EXECUTION_ID}`}
        >
          {children}
        </NuqsTestingAdapter>
      </QueryClientProvider>
    );
  }
  return renderHook(
    () => {
      useFlow();
      useFlowRealtime();
    },
    { wrapper: Wrapper },
  );
}

function nodeStatuses() {
  return Object.fromEntries(
    useNodeStore.getState().nodes.map((node) => [node.id, node.data.status]),
  );
}

async function acknowledgeSubscribe() {
  await waitFor(() => expect(socket.state.subscribes).toEqual([EXECUTION_ID]));
  const requestsBefore = executionRequests;
  act(() => socket.send("subscribe_graph_execution"));
  // The acknowledgement refetches once.
  await waitFor(() =>
    expect(executionRequests).toBeGreaterThan(requestsBefore),
  );
}

async function requestsDuring(ms: number) {
  const before = executionRequests;
  await act(() => new Promise((resolve) => setTimeout(resolve, ms)));
  return executionRequests - before;
}

describe("builder execution updates", () => {
  beforeEach(() => {
    socket.reset();
    run = { status: "RUNNING", nodes: { "node-1": "RUNNING" } };
    executionRequests = 0;
    heldResponses = null;
    server.use(
      http.get(`/api/proxy/api/graphs/${GRAPH_ID}`, () =>
        HttpResponse.json(graphResponse()),
      ),
      http.get(
        `/api/proxy/api/graphs/${GRAPH_ID}/executions/${EXECUTION_ID}`,
        async () => {
          executionRequests++;
          const snapshot = executionResponse();
          if (heldResponses) await heldResponses;
          return HttpResponse.json(snapshot);
        },
      ),
      http.get("/api/proxy/api/graphs", () => HttpResponse.json([])),
      http.get("/api/proxy/api/builder/blocks/batch", () =>
        HttpResponse.json([
          {
            id: BLOCK_ID,
            name: "Store Value",
            description: "",
            inputSchema: { type: "object", properties: {} },
            outputSchema: { type: "object", properties: {} },
            costs: [],
            categories: [],
            contributors: [],
            staticOutput: false,
            uiType: "Standard",
          },
        ]),
      ),
    );
  });

  afterEach(() => {
    useNodeStore.getState().setNodes([]);
    useEdgeStore.getState().setEdges([]);
    useGraphStore.getState().reset();
  });

  it("ends with every node's final status when the run finishes before the subscribe is acknowledged", async () => {
    renderBuilder();
    await waitFor(() => expect(nodeStatuses()["node-1"]).toBe("RUNNING"));

    act(() => socket.connect());
    await waitFor(() =>
      expect(socket.state.subscribes).toEqual([EXECUTION_ID]),
    );
    // The dry-run finishes while the subscription is still being set up, so
    // none of its node events ever reach the websocket.
    run = {
      status: "COMPLETED",
      nodes: {
        "node-1": "COMPLETED",
        "node-2": "COMPLETED",
        "node-3": "COMPLETED",
      },
    };
    act(() => socket.send("subscribe_graph_execution"));

    await waitFor(() =>
      expect(nodeStatuses()).toEqual({
        "node-1": "COMPLETED",
        "node-2": "COMPLETED",
        "node-3": "COMPLETED",
      }),
    );
    expect(useGraphStore.getState().isGraphRunning).toBe(false);
    expect(useNodeStore.getState().getLatestNodeOutputData("node-3")).toEqual({
      output: ["node-3"],
    });
  }, 15_000);

  it("polls while running until the subscribe is acknowledged, then only listens", async () => {
    renderBuilder();
    await waitFor(() => expect(nodeStatuses()["node-1"]).toBe("RUNNING"));
    act(() => socket.connect());

    // Subscribe sent but not acknowledged yet: keep polling.
    expect(await requestsDuring(POLL_INTERVAL_MS * 2.5)).toBeGreaterThan(0);

    await acknowledgeSubscribe();
    expect(await requestsDuring(POLL_INTERVAL_MS * 2.5)).toBe(0);

    act(() =>
      socket.send("node_execution_event", {
        ...nodeExecution("node-2", "RUNNING"),
        add_time: new Date("2026-01-05T09:00:00.000Z"),
      }),
    );
    expect(nodeStatuses()["node-2"]).toBe("RUNNING");
    expect(useEdgeStore.getState().edges[0].data?.beadUp).toBe(1);
  }, 20_000);

  it("polls again after the websocket drops until it resubscribes", async () => {
    renderBuilder();
    await waitFor(() => expect(nodeStatuses()["node-1"]).toBe("RUNNING"));
    act(() => socket.connect());
    await acknowledgeSubscribe();

    act(() => socket.disconnect());
    expect(await requestsDuring(POLL_INTERVAL_MS * 2.5)).toBeGreaterThan(0);

    socket.state.subscribes = [];
    act(() => socket.connect());
    await acknowledgeSubscribe();
    expect(await requestsDuring(POLL_INTERVAL_MS * 2.5)).toBe(0);
  }, 20_000);

  it("keeps newer websocket updates when a refetch that was in flight lands after them", async () => {
    renderBuilder();
    await waitFor(() => expect(nodeStatuses()["node-1"]).toBe("RUNNING"));
    act(() => socket.connect());
    await waitFor(() =>
      expect(socket.state.subscribes).toEqual([EXECUTION_ID]),
    );

    let release: () => void = () => undefined;
    heldResponses = new Promise((resolve) => {
      release = resolve;
    });
    const requestsBefore = executionRequests;
    act(() => socket.send("subscribe_graph_execution"));
    await waitFor(() =>
      expect(executionRequests).toBeGreaterThan(requestsBefore),
    );

    // The run finishes while the refetch, read as RUNNING, is on its way.
    run = { status: "COMPLETED", nodes: { "node-1": "COMPLETED" } };
    act(() => {
      socket.send("node_execution_event", nodeExecution("node-1", "COMPLETED"));
      socket.send("graph_execution_event", {
        id: EXECUTION_ID,
        status: "COMPLETED",
      });
    });
    expect(nodeStatuses()["node-1"]).toBe("COMPLETED");

    const requestsAtRelease = executionRequests;
    act(() => release());
    await act(() => new Promise((resolve) => setTimeout(resolve, 200)));

    expect(nodeStatuses()["node-1"]).toBe("COMPLETED");
    expect(useGraphStore.getState().isGraphRunning).toBe(false);
    expect(useNodeStore.getState().getLatestNodeOutputData("node-1")).toEqual({
      output: ["node-1"],
    });
    expect(executionRequests).toBe(requestsAtRelease);
  }, 15_000);

  it(
    "still catches up, slowly, on a run whose last events never reached the websocket",
    async () => {
      renderBuilder();
      await waitFor(() => expect(nodeStatuses()["node-1"]).toBe("RUNNING"));
      act(() => socket.connect());
      await acknowledgeSubscribe();
      act(() =>
        socket.send("node_execution_event", nodeExecution("node-1", "RUNNING")),
      );

      // The run finishes, but its events are lost before the websocket.
      run = { status: "COMPLETED", nodes: { "node-1": "COMPLETED" } };
      expect(await requestsDuring(POLL_INTERVAL_MS * 2.5)).toBe(0);

      await waitFor(() => expect(nodeStatuses()["node-1"]).toBe("COMPLETED"), {
        timeout: RECONCILE_INTERVAL_MS + 2_000,
      });
      expect(useGraphStore.getState().isGraphRunning).toBe(false);
    },
    RECONCILE_INTERVAL_MS + 15_000,
  );

  it("keeps the latest execution's status for a node that ran twice", async () => {
    run = { status: "RUNNING", nodes: { "node-1": "COMPLETED" } };
    renderBuilder();
    await waitFor(() => expect(nodeStatuses()["node-1"]).toBe("COMPLETED"));
    act(() => socket.connect());
    await acknowledgeSubscribe();

    const secondRun = {
      ...nodeExecution("node-1", "RUNNING"),
      node_exec_id: "exec-node-1-again",
    };
    act(() => socket.send("node_execution_event", secondRun));
    expect(nodeStatuses()["node-1"]).toBe("RUNNING");

    // A later snapshot lists both executions, the finished one first.
    run = {
      status: "RUNNING",
      nodes: {},
      earlierExecutions: [nodeExecution("node-1", "COMPLETED"), secondRun],
    };
    await acknowledgeSubscribe();
    await act(() => new Promise((resolve) => setTimeout(resolve, 200)));

    expect(nodeStatuses()["node-1"]).toBe("RUNNING");
    expect(
      useNodeStore.getState().getNodeExecutionResults("node-1"),
    ).toHaveLength(2);
  });

  it("refetches on every acknowledgement, so a late one for an earlier run cannot hide this run's", async () => {
    renderBuilder();
    await waitFor(() => expect(nodeStatuses()["node-1"]).toBe("RUNNING"));
    act(() => socket.connect());
    await waitFor(() =>
      expect(socket.state.subscribes).toEqual([EXECUTION_ID]),
    );

    // An ack left over from a run opened earlier arrives first...
    await acknowledgeSubscribe();
    // ...and the run finishes before this run's own subscription is in place.
    run = { status: "COMPLETED", nodes: { "node-1": "COMPLETED" } };
    await acknowledgeSubscribe();

    await waitFor(() => expect(nodeStatuses()["node-1"]).toBe("COMPLETED"));
    expect(useGraphStore.getState().isGraphRunning).toBe(false);
  });

  it("ignores a repeated websocket event", async () => {
    renderBuilder();
    await waitFor(() => expect(nodeStatuses()["node-1"]).toBe("RUNNING"));
    act(() => socket.connect());
    await acknowledgeSubscribe();

    const event = nodeExecution("node-2", "COMPLETED");
    act(() => socket.send("node_execution_event", event));
    const nodes = useNodeStore.getState().nodes;
    const edges = useEdgeStore.getState().edges;

    act(() => socket.send("node_execution_event", { ...event }));

    expect(useNodeStore.getState().nodes).toBe(nodes);
    expect(useEdgeStore.getState().edges).toBe(edges);
    expect(nodeStatuses()["node-2"]).toBe("COMPLETED");
  });

  it("ignores events for another run", async () => {
    renderBuilder();
    await waitFor(() => expect(nodeStatuses()["node-1"]).toBe("RUNNING"));
    act(() => socket.connect());
    await acknowledgeSubscribe();

    act(() =>
      socket.send("node_execution_event", {
        ...nodeExecution("node-3", "COMPLETED"),
        graph_exec_id: "another-execution",
      }),
    );

    expect(nodeStatuses()["node-3"]).toBeUndefined();
  });
});
