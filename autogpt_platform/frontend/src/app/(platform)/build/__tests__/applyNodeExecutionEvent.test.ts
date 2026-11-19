import { beforeEach, describe, expect, it } from "vitest";
import { useNodeStore } from "../stores/nodeStore";
import { useEdgeStore } from "../stores/edgeStore";
import { BlockUIType } from "../components/types";
import type { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import type { CustomEdge } from "../components/FlowEditor/edges/CustomEdge";
import type { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";

function createNode(id: string): CustomNode {
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues: {},
      title: "Store Value",
      description: "",
      inputSchema: {},
      outputSchema: {},
      uiType: BlockUIType.STANDARD,
      block_id: "block-1",
      costs: [],
      categories: [],
    },
  };
}

function createEdge(source: string, target: string): CustomEdge {
  return {
    id: `${source}->${target}`,
    source,
    target,
    sourceHandle: "output",
    targetHandle: "input",
    type: "custom",
  };
}

// A fresh object every call, with fresh Dates, like a poll response or a
// websocket message.
function createEvent(
  overrides: Partial<NodeExecutionResult> = {},
): NodeExecutionResult {
  return {
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    graph_exec_id: "graph-exec-1",
    node_exec_id: "node-exec-1",
    node_id: "1",
    block_id: "block-1",
    status: "RUNNING",
    input_data: { input: "hello" },
    output_data: {},
    add_time: new Date("2026-01-05T09:00:00.000Z"),
    queue_time: new Date("2026-01-05T09:00:00.000Z"),
    start_time: new Date("2026-01-05T09:00:01.000Z"),
    end_time: null,
    ...overrides,
  };
}

function countWrites() {
  const writes = { nodes: 0, edges: 0 };
  const stopNodes = useNodeStore.subscribe(() => writes.nodes++);
  const stopEdges = useEdgeStore.subscribe(() => writes.edges++);
  return {
    writes,
    stop() {
      stopNodes();
      stopEdges();
    },
  };
}

describe("applyNodeExecutionEvent", () => {
  beforeEach(() => {
    useNodeStore.setState({
      nodes: [createNode("0"), createNode("1")],
      latestNodeInputData: {},
      latestNodeOutputData: {},
      accumulatedNodeInputData: {},
      accumulatedNodeOutputData: {},
    });
    useEdgeStore.setState({ edges: [createEdge("0", "1")] });
  });

  it("updates status, results and edge beads with one write per store", () => {
    const counter = countWrites();

    useNodeStore.getState().applyNodeExecutionEvent("1", createEvent());
    counter.stop();

    expect(counter.writes).toEqual({ nodes: 1, edges: 1 });
    const state = useNodeStore.getState();
    expect(state.getNodeStatus("1")).toBe("RUNNING");
    expect(state.getNodeExecutionResults("1")).toHaveLength(1);
    expect(state.getLatestNodeInputData("1")).toEqual({ input: "hello" });
    expect(state.getAccumulatedNodeInputData("1")).toEqual({
      input: ["hello"],
    });
    expect(useEdgeStore.getState().edges[0].data).toMatchObject({
      beadUp: 1,
      beadDown: 1,
    });
  });

  it("returns the same state objects for an identical event", () => {
    useNodeStore.getState().applyNodeExecutionEvent("1", createEvent());
    const nodeState = useNodeStore.getState();
    const edgeState = useEdgeStore.getState();
    const counter = countWrites();

    useNodeStore.getState().applyNodeExecutionEvent("1", createEvent());
    counter.stop();

    expect(useNodeStore.getState()).toBe(nodeState);
    expect(useEdgeStore.getState()).toBe(edgeState);
    expect(counter.writes).toEqual({ nodes: 0, edges: 0 });
  });

  it("updates only the status when the payload is unchanged", () => {
    useNodeStore.getState().applyNodeExecutionEvent("1", createEvent());
    const [firstResult] = useNodeStore.getState().getNodeExecutionResults("1");
    const otherNode = useNodeStore.getState().nodes[0];

    useNodeStore
      .getState()
      .applyNodeExecutionEvent("1", createEvent({ status: "COMPLETED" }));

    const state = useNodeStore.getState();
    expect(state.getNodeStatus("1")).toBe("COMPLETED");
    expect(state.getNodeExecutionResults("1")).toEqual([firstResult]);
    expect(state.getNodeExecutionResults("1")[0]).toBe(firstResult);
    expect(state.nodes[0]).toBe(otherNode);
  });

  it("replaces a result whose output changed and rebuilds its accumulated data", () => {
    useNodeStore.getState().applyNodeExecutionEvent("1", createEvent());
    useNodeStore.getState().applyNodeExecutionEvent(
      "1",
      createEvent({
        status: "COMPLETED",
        output_data: { output: ["hello"] },
        end_time: new Date("2026-01-05T09:00:02.000Z"),
      }),
    );

    const state = useNodeStore.getState();
    expect(state.getNodeStatus("1")).toBe("COMPLETED");
    expect(state.getNodeExecutionResults("1")).toHaveLength(1);
    expect(state.getLatestNodeOutputData("1")).toEqual({ output: ["hello"] });
    expect(state.getAccumulatedNodeOutputData("1")).toEqual({
      output: ["hello"],
    });
    expect(state.getAccumulatedNodeInputData("1")).toEqual({
      input: ["hello"],
    });
  });

  it("appends a new execution of the same node", () => {
    useNodeStore.getState().applyNodeExecutionEvent("1", createEvent());
    useNodeStore.getState().applyNodeExecutionEvent(
      "1",
      createEvent({
        node_exec_id: "node-exec-2",
        input_data: { input: "again" },
      }),
    );

    const state = useNodeStore.getState();
    expect(state.getNodeExecutionResults("1")).toHaveLength(2);
    expect(state.getLatestNodeInputData("1")).toEqual({ input: "again" });
    expect(state.getAccumulatedNodeInputData("1")).toEqual({
      input: ["hello", "again"],
    });
  });

  it("leaves the store untouched for a node that is not on the canvas", () => {
    const nodeState = useNodeStore.getState();

    useNodeStore
      .getState()
      .applyNodeExecutionEvent("missing", createEvent({ node_id: "missing" }));

    expect(useNodeStore.getState()).toBe(nodeState);
  });
});

describe("updateNodeStatus", () => {
  beforeEach(() => {
    useNodeStore.setState({ nodes: [createNode("1")] });
  });

  it("leaves the store untouched when the status is unchanged", () => {
    useNodeStore.getState().updateNodeStatus("1", "RUNNING");
    const nodeState = useNodeStore.getState();

    useNodeStore.getState().updateNodeStatus("1", "RUNNING");

    expect(useNodeStore.getState()).toBe(nodeState);
  });
});

describe("updateNodeExecutionResult", () => {
  beforeEach(() => {
    useNodeStore.setState({ nodes: [createNode("1")] });
  });

  it("leaves the store untouched for a repeated result", () => {
    useNodeStore.getState().updateNodeExecutionResult("1", createEvent());
    const nodeState = useNodeStore.getState();

    useNodeStore.getState().updateNodeExecutionResult("1", createEvent());

    expect(useNodeStore.getState()).toBe(nodeState);
  });
});
