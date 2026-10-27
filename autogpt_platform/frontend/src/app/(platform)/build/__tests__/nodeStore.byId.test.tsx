import { Profiler } from "react";
import { act, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useNodeStore } from "../stores/nodeStore";
import { useEdgeStore } from "../stores/edgeStore";
import { useHistoryStore } from "../stores/historyStore";
import { BlockUIType } from "../components/types";
import { NodeContainer } from "../components/FlowEditor/nodes/CustomNode/components/NodeContainer";
import type { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import type { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";

function createNode(id: string, title = "Store Value"): CustomNode {
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues: { input: `value ${id}` },
      title,
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

function createResult(
  nodeId: string,
  status: NodeExecutionResult["status"],
): NodeExecutionResult {
  return {
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    graph_exec_id: "graph-exec-1",
    node_exec_id: `exec-${nodeId}`,
    node_id: nodeId,
    block_id: "block-1",
    status,
    input_data: {},
    output_data: { output: [`out ${nodeId}`] },
    add_time: new Date("2026-01-05T09:00:00.000Z"),
    queue_time: new Date("2026-01-05T09:00:00.000Z"),
    start_time: new Date("2026-01-05T09:00:01.000Z"),
    end_time: null,
  };
}

beforeEach(() => {
  useNodeStore.setState({
    nodes: [],
    latestNodeInputData: {},
    latestNodeOutputData: {},
    accumulatedNodeInputData: {},
    accumulatedNodeOutputData: {},
  });
  useEdgeStore.setState({ edges: [] });
  useHistoryStore.setState({ past: [], future: [] });
});

describe("nodeStore lookups by id", () => {
  it("does not scan the nodes array for each node's read", () => {
    const nodes = Array.from({ length: 50 }, (_, index) =>
      createNode(String(index)),
    );
    const findSpy = vi.spyOn(nodes, "find");
    const findIndexSpy = vi.spyOn(nodes, "findIndex");
    useNodeStore.getState().setNodes(nodes);

    const { getNodeStatus, getNodeErrors, getHardCodedValues } =
      useNodeStore.getState();
    for (const node of nodes) {
      getNodeStatus(node.id);
      getNodeErrors(node.id);
      expect(getHardCodedValues(node.id)).toEqual({
        input: `value ${node.id}`,
      });
    }

    expect(findSpy).not.toHaveBeenCalled();
    expect(findIndexSpy).not.toHaveBeenCalled();
  });

  it("reads the current values after every kind of write", () => {
    const store = useNodeStore.getState();
    store.setNodes([createNode("1"), createNode("2")]);
    expect(store.getNodeStatus("1")).toBeUndefined();

    store.applyNodeExecutionEvent("1", createResult("1", "RUNNING"));
    expect(useNodeStore.getState().getNodeStatus("1")).toBe("RUNNING");
    expect(useNodeStore.getState().getNodeExecutionResults("1")).toHaveLength(
      1,
    );
    expect(
      useNodeStore.getState().getLatestNodeExecutionResult("1")?.status,
    ).toBe("RUNNING");

    store.updateNodeErrors("2", { input: "Required" });
    expect(useNodeStore.getState().getNodeErrors("2")).toEqual({
      input: "Required",
    });
    store.clearNodeErrors("2");
    expect(useNodeStore.getState().getNodeErrors("2")).toBeUndefined();

    store.updateNodeData("2", { hardcodedValues: { input: "edited" } });
    expect(useNodeStore.getState().getHardCodedValues("2")).toEqual({
      input: "edited",
    });

    store.cleanNodesStatuses();
    expect(useNodeStore.getState().getNodeStatus("1")).toBeUndefined();

    store.onNodesChange([{ type: "remove", id: "1" }]);
    expect(useNodeStore.getState().getNodeStatus("1")).toBeUndefined();
    expect(useNodeStore.getState().getHardCodedValues("1")).toEqual({});
    expect(useNodeStore.getState().getNodeExecutionResults("1")).toEqual([]);
  });

  it("reads a node the caller replaced in an array it sets again", () => {
    const nodes = [createNode("1")];
    useNodeStore.getState().setNodes(nodes);
    expect(useNodeStore.getState().getNodeStatus("1")).toBeUndefined();

    const replaced = createNode("1");
    replaced.data.status = "COMPLETED";
    nodes[0] = replaced;
    useNodeStore.getState().setNodes(nodes);

    expect(useNodeStore.getState().getNodeStatus("1")).toBe("COMPLETED");
  });

  it("returns the first node when two share an id, as find did", () => {
    const first = createNode("1", "First");
    const second = createNode("1", "Second");
    first.data.status = "COMPLETED";
    second.data.status = "FAILED";
    useNodeStore.getState().setNodes([first, second]);

    expect(useNodeStore.getState().getNodeStatus("1")).toBe("COMPLETED");
  });
});

describe("NodeContainer", () => {
  it("re-renders only the node whose status changed", () => {
    useNodeStore.getState().setNodes([createNode("A"), createNode("B")]);
    const renders: Record<string, number> = { A: 0, B: 0 };

    render(
      <>
        {["A", "B"].map((nodeId) => (
          <Profiler
            key={nodeId}
            id={nodeId}
            onRender={() => {
              renders[nodeId] += 1;
            }}
          >
            <NodeContainer nodeId={nodeId} selected={false}>
              <span>{`node ${nodeId}`}</span>
            </NodeContainer>
          </Profiler>
        ))}
      </>,
    );
    expect(renders).toEqual({ A: 1, B: 1 });

    act(() => {
      useNodeStore.getState().updateNodeStatus("B", "RUNNING");
    });

    expect(renders).toEqual({ A: 1, B: 2 });
    expect(screen.getByText("node B").parentElement?.className).toContain(
      "ring-amber-300",
    );
    expect(screen.getByText("node A").parentElement?.className).not.toContain(
      "ring-amber-300",
    );
  });
});
