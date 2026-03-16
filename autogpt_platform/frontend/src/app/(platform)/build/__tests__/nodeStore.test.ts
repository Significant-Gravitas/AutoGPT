import { describe, it, expect, beforeEach, vi } from "vitest";
import { useNodeStore } from "../stores/nodeStore";
import { useHistoryStore } from "../stores/historyStore";
import { useEdgeStore } from "../stores/edgeStore";
import { BlockUIType } from "../components/types";
import type { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import type { CustomNodeData } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import type { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";

function createTestNode(overrides: {
  id: string;
  position?: { x: number; y: number };
  data?: Partial<CustomNodeData>;
}): CustomNode {
  const defaults: CustomNodeData = {
    hardcodedValues: {},
    title: "Test Block",
    description: "A test block",
    inputSchema: {},
    outputSchema: {},
    uiType: BlockUIType.STANDARD,
    block_id: "test-block-id",
    costs: [],
    categories: [],
  };

  return {
    id: overrides.id,
    type: "custom",
    position: overrides.position ?? { x: 0, y: 0 },
    data: { ...defaults, ...overrides.data },
  };
}

function createExecutionResult(
  overrides: Partial<NodeExecutionResult> = {},
): NodeExecutionResult {
  return {
    node_exec_id: overrides.node_exec_id ?? "exec-1",
    node_id: overrides.node_id ?? "1",
    graph_exec_id: overrides.graph_exec_id ?? "graph-exec-1",
    graph_id: overrides.graph_id ?? "graph-1",
    graph_version: overrides.graph_version ?? 1,
    user_id: overrides.user_id ?? "test-user",
    block_id: overrides.block_id ?? "block-1",
    status: overrides.status ?? "COMPLETED",
    input_data: overrides.input_data ?? { input_key: "input_value" },
    output_data: overrides.output_data ?? { output_key: ["output_value"] },
    add_time: overrides.add_time ?? new Date("2024-01-01T00:00:00Z"),
    queue_time: overrides.queue_time ?? new Date("2024-01-01T00:00:00Z"),
    start_time: overrides.start_time ?? new Date("2024-01-01T00:00:01Z"),
    end_time: overrides.end_time ?? new Date("2024-01-01T00:00:02Z"),
  };
}

function resetStores() {
  useNodeStore.setState({
    nodes: [],
    nodeCounter: 0,
    nodeAdvancedStates: {},
    latestNodeInputData: {},
    latestNodeOutputData: {},
    accumulatedNodeInputData: {},
    accumulatedNodeOutputData: {},
    nodesInResolutionMode: new Set(),
    brokenEdgeIDs: new Map(),
    nodeResolutionData: new Map(),
  });
  useEdgeStore.setState({ edges: [] });
  useHistoryStore.setState({ past: [], future: [] });
}

describe("nodeStore", () => {
  beforeEach(() => {
    resetStores();
    vi.restoreAllMocks();
  });

  describe("node lifecycle", () => {
    it("starts with empty nodes", () => {
      const { nodes } = useNodeStore.getState();
      expect(nodes).toEqual([]);
    });

    it("adds a single node with addNode", () => {
      const node = createTestNode({ id: "1" });
      useNodeStore.getState().addNode(node);

      const { nodes } = useNodeStore.getState();
      expect(nodes).toHaveLength(1);
      expect(nodes[0].id).toBe("1");
    });

    it("sets nodes with setNodes, replacing existing ones", () => {
      const node1 = createTestNode({ id: "1" });
      const node2 = createTestNode({ id: "2" });
      useNodeStore.getState().addNode(node1);

      useNodeStore.getState().setNodes([node2]);

      const { nodes } = useNodeStore.getState();
      expect(nodes).toHaveLength(1);
      expect(nodes[0].id).toBe("2");
    });

    it("removes nodes via onNodesChange", () => {
      const node = createTestNode({ id: "1" });
      useNodeStore.getState().setNodes([node]);

      useNodeStore.getState().onNodesChange([{ type: "remove", id: "1" }]);

      expect(useNodeStore.getState().nodes).toHaveLength(0);
    });

    it("updates node data with updateNodeData", () => {
      const node = createTestNode({ id: "1" });
      useNodeStore.getState().addNode(node);

      useNodeStore.getState().updateNodeData("1", { title: "Updated Title" });

      const updated = useNodeStore.getState().nodes[0];
      expect(updated.data.title).toBe("Updated Title");
      expect(updated.data.block_id).toBe("test-block-id");
    });

    it("updateNodeData does not affect other nodes", () => {
      const node1 = createTestNode({ id: "1" });
      const node2 = createTestNode({
        id: "2",
        data: { title: "Node 2" },
      });
      useNodeStore.getState().setNodes([node1, node2]);

      useNodeStore.getState().updateNodeData("1", { title: "Changed" });

      expect(useNodeStore.getState().nodes[1].data.title).toBe("Node 2");
    });
  });

  describe("bulk operations", () => {
    it("adds multiple nodes with addNodes", () => {
      const nodes = [
        createTestNode({ id: "1" }),
        createTestNode({ id: "2" }),
        createTestNode({ id: "3" }),
      ];
      useNodeStore.getState().addNodes(nodes);

      expect(useNodeStore.getState().nodes).toHaveLength(3);
    });

    it("removes multiple nodes via onNodesChange", () => {
      const nodes = [
        createTestNode({ id: "1" }),
        createTestNode({ id: "2" }),
        createTestNode({ id: "3" }),
      ];
      useNodeStore.getState().setNodes(nodes);

      useNodeStore.getState().onNodesChange([
        { type: "remove", id: "1" },
        { type: "remove", id: "3" },
      ]);

      const remaining = useNodeStore.getState().nodes;
      expect(remaining).toHaveLength(1);
      expect(remaining[0].id).toBe("2");
    });
  });

  describe("nodeCounter", () => {
    it("starts at zero", () => {
      expect(useNodeStore.getState().nodeCounter).toBe(0);
    });

    it("increments the counter", () => {
      useNodeStore.getState().incrementNodeCounter();
      expect(useNodeStore.getState().nodeCounter).toBe(1);

      useNodeStore.getState().incrementNodeCounter();
      expect(useNodeStore.getState().nodeCounter).toBe(2);
    });

    it("sets the counter to a specific value", () => {
      useNodeStore.getState().setNodeCounter(42);
      expect(useNodeStore.getState().nodeCounter).toBe(42);
    });
  });

  describe("advanced states", () => {
    it("defaults to false for unknown node IDs", () => {
      expect(useNodeStore.getState().getShowAdvanced("unknown")).toBe(false);
    });

    it("toggles advanced state", () => {
      useNodeStore.getState().toggleAdvanced("node-1");
      expect(useNodeStore.getState().getShowAdvanced("node-1")).toBe(true);

      useNodeStore.getState().toggleAdvanced("node-1");
      expect(useNodeStore.getState().getShowAdvanced("node-1")).toBe(false);
    });

    it("sets advanced state explicitly", () => {
      useNodeStore.getState().setShowAdvanced("node-1", true);
      expect(useNodeStore.getState().getShowAdvanced("node-1")).toBe(true);

      useNodeStore.getState().setShowAdvanced("node-1", false);
      expect(useNodeStore.getState().getShowAdvanced("node-1")).toBe(false);
    });
  });

  describe("convertCustomNodeToBackendNode", () => {
    it("converts a node with minimal data", () => {
      const node = createTestNode({
        id: "42",
        position: { x: 100, y: 200 },
      });

      const backend = useNodeStore
        .getState()
        .convertCustomNodeToBackendNode(node);

      expect(backend.id).toBe("42");
      expect(backend.block_id).toBe("test-block-id");
      expect(backend.input_default).toEqual({});
      expect(backend.metadata).toEqual({ position: { x: 100, y: 200 } });
    });

    it("includes customized_name when present in metadata", () => {
      const node = createTestNode({
        id: "1",
        data: {
          metadata: { customized_name: "My Custom Name" },
        },
      });

      const backend = useNodeStore
        .getState()
        .convertCustomNodeToBackendNode(node);

      expect(backend.metadata).toHaveProperty(
        "customized_name",
        "My Custom Name",
      );
    });

    it("includes credentials_optional when present in metadata", () => {
      const node = createTestNode({
        id: "1",
        data: {
          metadata: { credentials_optional: true },
        },
      });

      const backend = useNodeStore
        .getState()
        .convertCustomNodeToBackendNode(node);

      expect(backend.metadata).toHaveProperty("credentials_optional", true);
    });

    it("prunes empty values from hardcodedValues", () => {
      const node = createTestNode({
        id: "1",
        data: {
          hardcodedValues: { filled: "value", empty: "" },
        },
      });

      const backend = useNodeStore
        .getState()
        .convertCustomNodeToBackendNode(node);

      expect(backend.input_default).toEqual({ filled: "value" });
      expect(backend.input_default).not.toHaveProperty("empty");
    });
  });

  describe("getBackendNodes", () => {
    it("converts all nodes to backend format", () => {
      useNodeStore
        .getState()
        .setNodes([
          createTestNode({ id: "1", position: { x: 0, y: 0 } }),
          createTestNode({ id: "2", position: { x: 100, y: 100 } }),
        ]);

      const backendNodes = useNodeStore.getState().getBackendNodes();

      expect(backendNodes).toHaveLength(2);
      expect(backendNodes[0].id).toBe("1");
      expect(backendNodes[1].id).toBe("2");
    });
  });

  describe("node status", () => {
    it("returns undefined for a node with no status", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));
      expect(useNodeStore.getState().getNodeStatus("1")).toBeUndefined();
    });

    it("updates node status", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));

      useNodeStore.getState().updateNodeStatus("1", "RUNNING");
      expect(useNodeStore.getState().getNodeStatus("1")).toBe("RUNNING");

      useNodeStore.getState().updateNodeStatus("1", "COMPLETED");
      expect(useNodeStore.getState().getNodeStatus("1")).toBe("COMPLETED");
    });

    it("cleans all node statuses", () => {
      useNodeStore
        .getState()
        .setNodes([createTestNode({ id: "1" }), createTestNode({ id: "2" })]);
      useNodeStore.getState().updateNodeStatus("1", "RUNNING");
      useNodeStore.getState().updateNodeStatus("2", "COMPLETED");

      useNodeStore.getState().cleanNodesStatuses();

      expect(useNodeStore.getState().getNodeStatus("1")).toBeUndefined();
      expect(useNodeStore.getState().getNodeStatus("2")).toBeUndefined();
    });

    it("updating status for non-existent node does not crash", () => {
      useNodeStore.getState().updateNodeStatus("nonexistent", "RUNNING");
      expect(
        useNodeStore.getState().getNodeStatus("nonexistent"),
      ).toBeUndefined();
    });
  });

  describe("execution result tracking", () => {
    it("returns empty array for node with no results", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));
      expect(useNodeStore.getState().getNodeExecutionResults("1")).toEqual([]);
    });

    it("tracks a single execution result", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));
      const result = createExecutionResult({ node_id: "1" });

      useNodeStore.getState().updateNodeExecutionResult("1", result);

      const results = useNodeStore.getState().getNodeExecutionResults("1");
      expect(results).toHaveLength(1);
      expect(results[0].node_exec_id).toBe("exec-1");
    });

    it("accumulates multiple execution results", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));

      useNodeStore.getState().updateNodeExecutionResult(
        "1",
        createExecutionResult({
          node_exec_id: "exec-1",
          input_data: { key: "val1" },
          output_data: { key: ["out1"] },
        }),
      );
      useNodeStore.getState().updateNodeExecutionResult(
        "1",
        createExecutionResult({
          node_exec_id: "exec-2",
          input_data: { key: "val2" },
          output_data: { key: ["out2"] },
        }),
      );

      expect(useNodeStore.getState().getNodeExecutionResults("1")).toHaveLength(
        2,
      );
    });

    it("updates latest input/output data", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));

      useNodeStore.getState().updateNodeExecutionResult(
        "1",
        createExecutionResult({
          node_exec_id: "exec-1",
          input_data: { key: "first" },
          output_data: { key: ["first_out"] },
        }),
      );
      useNodeStore.getState().updateNodeExecutionResult(
        "1",
        createExecutionResult({
          node_exec_id: "exec-2",
          input_data: { key: "second" },
          output_data: { key: ["second_out"] },
        }),
      );

      expect(useNodeStore.getState().getLatestNodeInputData("1")).toEqual({
        key: "second",
      });
      expect(useNodeStore.getState().getLatestNodeOutputData("1")).toEqual({
        key: ["second_out"],
      });
    });

    it("accumulates input/output data across results", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));

      useNodeStore.getState().updateNodeExecutionResult(
        "1",
        createExecutionResult({
          node_exec_id: "exec-1",
          input_data: { key: "val1" },
          output_data: { key: ["out1"] },
        }),
      );
      useNodeStore.getState().updateNodeExecutionResult(
        "1",
        createExecutionResult({
          node_exec_id: "exec-2",
          input_data: { key: "val2" },
          output_data: { key: ["out2"] },
        }),
      );

      const accInput = useNodeStore.getState().getAccumulatedNodeInputData("1");
      expect(accInput.key).toEqual(["val1", "val2"]);

      const accOutput = useNodeStore
        .getState()
        .getAccumulatedNodeOutputData("1");
      expect(accOutput.key).toEqual(["out1", "out2"]);
    });

    it("deduplicates execution results by node_exec_id", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));

      useNodeStore.getState().updateNodeExecutionResult(
        "1",
        createExecutionResult({
          node_exec_id: "exec-1",
          input_data: { key: "original" },
          output_data: { key: ["original_out"] },
        }),
      );
      useNodeStore.getState().updateNodeExecutionResult(
        "1",
        createExecutionResult({
          node_exec_id: "exec-1",
          input_data: { key: "updated" },
          output_data: { key: ["updated_out"] },
        }),
      );

      const results = useNodeStore.getState().getNodeExecutionResults("1");
      expect(results).toHaveLength(1);
      expect(results[0].input_data).toEqual({ key: "updated" });
    });

    it("returns the latest execution result", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));

      useNodeStore
        .getState()
        .updateNodeExecutionResult(
          "1",
          createExecutionResult({ node_exec_id: "exec-1" }),
        );
      useNodeStore
        .getState()
        .updateNodeExecutionResult(
          "1",
          createExecutionResult({ node_exec_id: "exec-2" }),
        );

      const latest = useNodeStore.getState().getLatestNodeExecutionResult("1");
      expect(latest?.node_exec_id).toBe("exec-2");
    });

    it("returns undefined for latest result on unknown node", () => {
      expect(
        useNodeStore.getState().getLatestNodeExecutionResult("unknown"),
      ).toBeUndefined();
    });

    it("clears all execution results", () => {
      useNodeStore
        .getState()
        .setNodes([createTestNode({ id: "1" }), createTestNode({ id: "2" })]);
      useNodeStore
        .getState()
        .updateNodeExecutionResult(
          "1",
          createExecutionResult({ node_exec_id: "exec-1" }),
        );
      useNodeStore
        .getState()
        .updateNodeExecutionResult(
          "2",
          createExecutionResult({ node_exec_id: "exec-2" }),
        );

      useNodeStore.getState().clearAllNodeExecutionResults();

      expect(useNodeStore.getState().getNodeExecutionResults("1")).toEqual([]);
      expect(useNodeStore.getState().getNodeExecutionResults("2")).toEqual([]);
      expect(
        useNodeStore.getState().getLatestNodeInputData("1"),
      ).toBeUndefined();
      expect(
        useNodeStore.getState().getLatestNodeOutputData("1"),
      ).toBeUndefined();
      expect(useNodeStore.getState().getAccumulatedNodeInputData("1")).toEqual(
        {},
      );
      expect(useNodeStore.getState().getAccumulatedNodeOutputData("1")).toEqual(
        {},
      );
    });

    it("returns empty object for accumulated data on unknown node", () => {
      expect(
        useNodeStore.getState().getAccumulatedNodeInputData("unknown"),
      ).toEqual({});
      expect(
        useNodeStore.getState().getAccumulatedNodeOutputData("unknown"),
      ).toEqual({});
    });
  });

  describe("getNodeBlockUIType", () => {
    it("returns the node UI type", () => {
      useNodeStore.getState().addNode(
        createTestNode({
          id: "1",
          data: {
            uiType: BlockUIType.INPUT,
          },
        }),
      );

      expect(useNodeStore.getState().getNodeBlockUIType("1")).toBe(
        BlockUIType.INPUT,
      );
    });

    it("defaults to STANDARD for unknown node IDs", () => {
      expect(useNodeStore.getState().getNodeBlockUIType("unknown")).toBe(
        BlockUIType.STANDARD,
      );
    });
  });

  describe("hasWebhookNodes", () => {
    it("returns false when there are no webhook nodes", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));
      expect(useNodeStore.getState().hasWebhookNodes()).toBe(false);
    });

    it("returns true when a WEBHOOK node exists", () => {
      useNodeStore.getState().addNode(
        createTestNode({
          id: "1",
          data: {
            uiType: BlockUIType.WEBHOOK,
          },
        }),
      );
      expect(useNodeStore.getState().hasWebhookNodes()).toBe(true);
    });

    it("returns true when a WEBHOOK_MANUAL node exists", () => {
      useNodeStore.getState().addNode(
        createTestNode({
          id: "1",
          data: {
            uiType: BlockUIType.WEBHOOK_MANUAL,
          },
        }),
      );
      expect(useNodeStore.getState().hasWebhookNodes()).toBe(true);
    });
  });

  describe("node errors", () => {
    it("returns undefined for a node with no errors", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));
      expect(useNodeStore.getState().getNodeErrors("1")).toBeUndefined();
    });

    it("sets and retrieves node errors", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));

      const errors = { field1: "required", field2: "invalid" };
      useNodeStore.getState().updateNodeErrors("1", errors);

      expect(useNodeStore.getState().getNodeErrors("1")).toEqual(errors);
    });

    it("clears errors for a specific node", () => {
      useNodeStore
        .getState()
        .setNodes([createTestNode({ id: "1" }), createTestNode({ id: "2" })]);
      useNodeStore.getState().updateNodeErrors("1", { f: "err" });
      useNodeStore.getState().updateNodeErrors("2", { g: "err2" });

      useNodeStore.getState().clearNodeErrors("1");

      expect(useNodeStore.getState().getNodeErrors("1")).toBeUndefined();
      expect(useNodeStore.getState().getNodeErrors("2")).toEqual({ g: "err2" });
    });

    it("clears all node errors", () => {
      useNodeStore
        .getState()
        .setNodes([createTestNode({ id: "1" }), createTestNode({ id: "2" })]);
      useNodeStore.getState().updateNodeErrors("1", { a: "err1" });
      useNodeStore.getState().updateNodeErrors("2", { b: "err2" });

      useNodeStore.getState().clearAllNodeErrors();

      expect(useNodeStore.getState().getNodeErrors("1")).toBeUndefined();
      expect(useNodeStore.getState().getNodeErrors("2")).toBeUndefined();
    });

    it("sets errors by backend ID matching node id", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "backend-1" }));

      useNodeStore
        .getState()
        .setNodeErrorsForBackendId("backend-1", { x: "error" });

      expect(useNodeStore.getState().getNodeErrors("backend-1")).toEqual({
        x: "error",
      });
    });
  });

  describe("getHardCodedValues", () => {
    it("returns hardcoded values for a node", () => {
      useNodeStore.getState().addNode(
        createTestNode({
          id: "1",
          data: {
            hardcodedValues: { key: "value" },
          },
        }),
      );

      expect(useNodeStore.getState().getHardCodedValues("1")).toEqual({
        key: "value",
      });
    });

    it("returns empty object for unknown node", () => {
      expect(useNodeStore.getState().getHardCodedValues("unknown")).toEqual({});
    });
  });

  describe("credentials optional", () => {
    it("sets credentials_optional in node metadata", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));

      useNodeStore.getState().setCredentialsOptional("1", true);

      const node = useNodeStore.getState().nodes[0];
      expect(node.data.metadata?.credentials_optional).toBe(true);
    });
  });

  describe("resolution mode", () => {
    it("defaults to not in resolution mode", () => {
      expect(useNodeStore.getState().isNodeInResolutionMode("1")).toBe(false);
    });

    it("enters and exits resolution mode", () => {
      useNodeStore.getState().setNodeResolutionMode("1", true);
      expect(useNodeStore.getState().isNodeInResolutionMode("1")).toBe(true);

      useNodeStore.getState().setNodeResolutionMode("1", false);
      expect(useNodeStore.getState().isNodeInResolutionMode("1")).toBe(false);
    });

    it("tracks broken edge IDs", () => {
      useNodeStore.getState().setBrokenEdgeIDs("node-1", ["edge-1", "edge-2"]);

      expect(useNodeStore.getState().isEdgeBroken("edge-1")).toBe(true);
      expect(useNodeStore.getState().isEdgeBroken("edge-2")).toBe(true);
      expect(useNodeStore.getState().isEdgeBroken("edge-3")).toBe(false);
    });

    it("removes individual broken edge IDs", () => {
      useNodeStore.getState().setBrokenEdgeIDs("node-1", ["edge-1", "edge-2"]);
      useNodeStore.getState().removeBrokenEdgeID("node-1", "edge-1");

      expect(useNodeStore.getState().isEdgeBroken("edge-1")).toBe(false);
      expect(useNodeStore.getState().isEdgeBroken("edge-2")).toBe(true);
    });

    it("clears all resolution state", () => {
      useNodeStore.getState().setNodeResolutionMode("1", true);
      useNodeStore.getState().setBrokenEdgeIDs("1", ["edge-1"]);

      useNodeStore.getState().clearResolutionState();

      expect(useNodeStore.getState().isNodeInResolutionMode("1")).toBe(false);
      expect(useNodeStore.getState().isEdgeBroken("edge-1")).toBe(false);
    });

    it("cleans up broken edges when exiting resolution mode", () => {
      useNodeStore.getState().setNodeResolutionMode("1", true);
      useNodeStore.getState().setBrokenEdgeIDs("1", ["edge-1"]);

      useNodeStore.getState().setNodeResolutionMode("1", false);

      expect(useNodeStore.getState().isEdgeBroken("edge-1")).toBe(false);
    });
  });

  describe("edge cases", () => {
    it("handles updating data on a non-existent node gracefully", () => {
      useNodeStore
        .getState()
        .updateNodeData("nonexistent", { title: "New Title" });

      expect(useNodeStore.getState().nodes).toHaveLength(0);
    });

    it("handles removing a non-existent node gracefully", () => {
      useNodeStore.getState().addNode(createTestNode({ id: "1" }));

      useNodeStore
        .getState()
        .onNodesChange([{ type: "remove", id: "nonexistent" }]);

      expect(useNodeStore.getState().nodes).toHaveLength(1);
    });

    it("handles duplicate node IDs in addNodes", () => {
      useNodeStore.getState().addNodes([
        createTestNode({
          id: "1",
          data: { title: "First" },
        }),
        createTestNode({
          id: "1",
          data: { title: "Second" },
        }),
      ]);

      const { nodes } = useNodeStore.getState();
      expect(nodes).toHaveLength(2);
      expect(nodes[0].data.title).toBe("First");
      expect(nodes[1].data.title).toBe("Second");
    });

    it("updating node status mid-execution preserves other data", () => {
      useNodeStore.getState().addNode(
        createTestNode({
          id: "1",
          data: {
            title: "My Node",
            hardcodedValues: { key: "val" },
          },
        }),
      );

      useNodeStore.getState().updateNodeStatus("1", "RUNNING");

      const node = useNodeStore.getState().nodes[0];
      expect(node.data.status).toBe("RUNNING");
      expect(node.data.title).toBe("My Node");
      expect(node.data.hardcodedValues).toEqual({ key: "val" });
    });

    it("execution result for non-existent node does not add it", () => {
      useNodeStore
        .getState()
        .updateNodeExecutionResult(
          "nonexistent",
          createExecutionResult({ node_exec_id: "exec-1" }),
        );

      expect(useNodeStore.getState().nodes).toHaveLength(0);
      expect(
        useNodeStore.getState().getNodeExecutionResults("nonexistent"),
      ).toEqual([]);
    });

    it("getBackendNodes returns empty array when no nodes exist", () => {
      expect(useNodeStore.getState().getBackendNodes()).toEqual([]);
    });
  });
});
