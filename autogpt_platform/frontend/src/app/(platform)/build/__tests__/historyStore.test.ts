import { describe, it, expect, beforeEach } from "vitest";
import { useHistoryStore } from "../stores/historyStore";
import { useNodeStore } from "../stores/nodeStore";
import { useEdgeStore } from "../stores/edgeStore";
import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import { CustomEdge } from "../components/FlowEditor/edges/CustomEdge";

function createTestNode(
  id: string,
  overrides: Partial<CustomNode> = {},
): CustomNode {
  return {
    id,
    type: "custom" as const,
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues: {},
      title: `Node ${id}`,
      description: "",
      inputSchema: {},
      outputSchema: {},
      uiType: "STANDARD" as never,
      block_id: `block-${id}`,
      costs: [],
      categories: [],
    },
    ...overrides,
  } as CustomNode;
}

function createTestEdge(
  id: string,
  source: string,
  target: string,
): CustomEdge {
  return {
    id,
    source,
    target,
    type: "custom" as const,
  } as CustomEdge;
}

async function flushMicrotasks() {
  await new Promise<void>((resolve) => queueMicrotask(resolve));
}

beforeEach(() => {
  useHistoryStore.getState().clear();
  useNodeStore.setState({ nodes: [] });
  useEdgeStore.setState({ edges: [] });
});

describe("historyStore", () => {
  describe("undo/redo single action", () => {
    it("undoes a single pushed state", async () => {
      const node = createTestNode("1");

      // Initialize history with node present as baseline
      useNodeStore.setState({ nodes: [node] });
      useHistoryStore.getState().initializeHistory();

      // Simulate a change: clear nodes
      useNodeStore.setState({ nodes: [] });

      // Undo should restore to [node]
      useHistoryStore.getState().undo();
      expect(useNodeStore.getState().nodes).toEqual([node]);
      expect(useHistoryStore.getState().future).toHaveLength(1);
      expect(useHistoryStore.getState().future[0].nodes).toEqual([]);
    });

    it("redoes after undo", async () => {
      const node = createTestNode("1");

      useNodeStore.setState({ nodes: [node] });
      useHistoryStore.getState().initializeHistory();

      // Change: clear nodes
      useNodeStore.setState({ nodes: [] });

      // Undo → back to [node]
      useHistoryStore.getState().undo();
      expect(useNodeStore.getState().nodes).toEqual([node]);

      // Redo → back to []
      useHistoryStore.getState().redo();
      expect(useNodeStore.getState().nodes).toEqual([]);
    });
  });

  describe("undo/redo multiple actions", () => {
    it("undoes through multiple states in order", async () => {
      const node1 = createTestNode("1");
      const node2 = createTestNode("2");
      const node3 = createTestNode("3");

      // Initialize with [node1] as baseline
      useNodeStore.setState({ nodes: [node1] });
      useHistoryStore.getState().initializeHistory();

      // Second change: add node2, push pre-change state
      useNodeStore.setState({ nodes: [node1, node2] });
      useHistoryStore.getState().pushState({ nodes: [node1], edges: [] });
      await flushMicrotasks();

      // Third change: add node3, push pre-change state
      useNodeStore.setState({ nodes: [node1, node2, node3] });
      useHistoryStore
        .getState()
        .pushState({ nodes: [node1, node2], edges: [] });
      await flushMicrotasks();

      // Undo 1: back to [node1, node2]
      useHistoryStore.getState().undo();
      expect(useNodeStore.getState().nodes).toEqual([node1, node2]);

      // Undo 2: back to [node1]
      useHistoryStore.getState().undo();
      expect(useNodeStore.getState().nodes).toEqual([node1]);
    });
  });

  describe("undo past empty history", () => {
    it("does nothing when there is no history to undo", () => {
      useHistoryStore.getState().undo();

      expect(useNodeStore.getState().nodes).toEqual([]);
      expect(useEdgeStore.getState().edges).toEqual([]);
      expect(useHistoryStore.getState().past).toHaveLength(1);
    });

    it("does nothing when current state equals last past entry", () => {
      expect(useHistoryStore.getState().canUndo()).toBe(false);

      useHistoryStore.getState().undo();

      expect(useHistoryStore.getState().past).toHaveLength(1);
      expect(useHistoryStore.getState().future).toHaveLength(0);
    });
  });

  describe("state consistency: undo after node add restores previous, redo restores added", () => {
    it("undo removes added node, redo restores it", async () => {
      const node = createTestNode("added");

      useNodeStore.setState({ nodes: [node] });
      useHistoryStore.getState().pushState({ nodes: [], edges: [] });
      await flushMicrotasks();

      useHistoryStore.getState().undo();
      expect(useNodeStore.getState().nodes).toEqual([]);

      useHistoryStore.getState().redo();
      expect(useNodeStore.getState().nodes).toEqual([node]);
    });
  });

  describe("history limits", () => {
    it("does not grow past MAX_HISTORY (50)", async () => {
      for (let i = 0; i < 60; i++) {
        const node = createTestNode(`node-${i}`);
        useNodeStore.setState({ nodes: [node] });
        useHistoryStore.getState().pushState({
          nodes: [createTestNode(`node-${i - 1}`)],
          edges: [],
        });
        await flushMicrotasks();
      }

      expect(useHistoryStore.getState().past.length).toBeLessThanOrEqual(50);
    });
  });

  describe("edge cases", () => {
    it("redo does nothing when future is empty", () => {
      const nodesBefore = useNodeStore.getState().nodes;
      const edgesBefore = useEdgeStore.getState().edges;

      useHistoryStore.getState().redo();

      expect(useNodeStore.getState().nodes).toEqual(nodesBefore);
      expect(useEdgeStore.getState().edges).toEqual(edgesBefore);
    });

    it("interleaved undo/redo sequence", async () => {
      const node1 = createTestNode("1");
      const node2 = createTestNode("2");
      const node3 = createTestNode("3");

      useNodeStore.setState({ nodes: [node1] });
      useHistoryStore.getState().pushState({ nodes: [], edges: [] });
      await flushMicrotasks();

      useNodeStore.setState({ nodes: [node1, node2] });
      useHistoryStore.getState().pushState({ nodes: [node1], edges: [] });
      await flushMicrotasks();

      useNodeStore.setState({ nodes: [node1, node2, node3] });
      useHistoryStore.getState().pushState({
        nodes: [node1, node2],
        edges: [],
      });
      await flushMicrotasks();

      useHistoryStore.getState().undo();
      expect(useNodeStore.getState().nodes).toEqual([node1, node2]);

      useHistoryStore.getState().undo();
      expect(useNodeStore.getState().nodes).toEqual([node1]);

      useHistoryStore.getState().redo();
      expect(useNodeStore.getState().nodes).toEqual([node1, node2]);

      useHistoryStore.getState().undo();
      expect(useNodeStore.getState().nodes).toEqual([node1]);

      useHistoryStore.getState().redo();
      useHistoryStore.getState().redo();
      expect(useNodeStore.getState().nodes).toEqual([node1, node2, node3]);
    });
  });

  describe("canUndo / canRedo", () => {
    it("canUndo is false on fresh store", () => {
      expect(useHistoryStore.getState().canUndo()).toBe(false);
    });

    it("canUndo is true when current state differs from last past entry", async () => {
      const node = createTestNode("1");
      useNodeStore.setState({ nodes: [node] });
      useHistoryStore.getState().pushState({ nodes: [], edges: [] });
      await flushMicrotasks();

      expect(useHistoryStore.getState().canUndo()).toBe(true);
    });

    it("canRedo is false on fresh store", () => {
      expect(useHistoryStore.getState().canRedo()).toBe(false);
    });

    it("canRedo is true after undo", async () => {
      const node = createTestNode("1");
      useNodeStore.setState({ nodes: [node] });
      useHistoryStore.getState().pushState({ nodes: [], edges: [] });
      await flushMicrotasks();

      useHistoryStore.getState().undo();

      expect(useHistoryStore.getState().canRedo()).toBe(true);
    });

    it("canRedo becomes false after redo exhausts future", async () => {
      const node = createTestNode("1");
      useNodeStore.setState({ nodes: [node] });
      useHistoryStore.getState().pushState({ nodes: [], edges: [] });
      await flushMicrotasks();

      useHistoryStore.getState().undo();
      useHistoryStore.getState().redo();

      expect(useHistoryStore.getState().canRedo()).toBe(false);
    });
  });

  describe("pushState deduplication", () => {
    it("does not push a state identical to the last past entry", async () => {
      useHistoryStore.getState().pushState({ nodes: [], edges: [] });
      await flushMicrotasks();

      expect(useHistoryStore.getState().past).toHaveLength(1);
    });

    it("does not push if state matches current node/edge store state", async () => {
      const node = createTestNode("1");
      useNodeStore.setState({ nodes: [node] });
      useEdgeStore.setState({ edges: [] });

      useHistoryStore.getState().pushState({ nodes: [node], edges: [] });
      await flushMicrotasks();

      expect(useHistoryStore.getState().past).toHaveLength(1);
    });
  });

  describe("initializeHistory", () => {
    it("resets history with current node/edge store state", async () => {
      const node = createTestNode("1");
      const edge = createTestEdge("e1", "1", "2");

      useNodeStore.setState({ nodes: [node] });
      useEdgeStore.setState({ edges: [edge] });

      useNodeStore.setState({ nodes: [node, createTestNode("2")] });
      useHistoryStore.getState().pushState({ nodes: [node], edges: [edge] });
      await flushMicrotasks();

      useHistoryStore.getState().initializeHistory();

      const { past, future } = useHistoryStore.getState();
      expect(past).toHaveLength(1);
      expect(past[0].nodes).toEqual(useNodeStore.getState().nodes);
      expect(past[0].edges).toEqual(useEdgeStore.getState().edges);
      expect(future).toHaveLength(0);
    });
  });

  describe("clear", () => {
    it("resets to empty initial state", async () => {
      const node = createTestNode("1");
      useNodeStore.setState({ nodes: [node] });
      useHistoryStore.getState().pushState({ nodes: [], edges: [] });
      await flushMicrotasks();

      useHistoryStore.getState().clear();

      const { past, future } = useHistoryStore.getState();
      expect(past).toEqual([{ nodes: [], edges: [] }]);
      expect(future).toEqual([]);
    });
  });

  describe("microtask batching", () => {
    it("only commits the first state when multiple pushState calls happen in the same tick", async () => {
      const node1 = createTestNode("1");
      const node2 = createTestNode("2");
      const node3 = createTestNode("3");

      useNodeStore.setState({ nodes: [node1, node2, node3] });

      useHistoryStore.getState().pushState({ nodes: [node1], edges: [] });
      useHistoryStore.getState().pushState({ nodes: [node2], edges: [] });
      useHistoryStore
        .getState()
        .pushState({ nodes: [node1, node2], edges: [] });
      await flushMicrotasks();

      const { past } = useHistoryStore.getState();
      expect(past).toHaveLength(2);
      expect(past[1].nodes).toEqual([node1]);
    });

    it("commits separately when pushState calls are in different ticks", async () => {
      const node1 = createTestNode("1");
      const node2 = createTestNode("2");

      useNodeStore.setState({ nodes: [node1, node2] });

      useHistoryStore.getState().pushState({ nodes: [node1], edges: [] });
      await flushMicrotasks();

      useHistoryStore.getState().pushState({ nodes: [node2], edges: [] });
      await flushMicrotasks();

      const { past } = useHistoryStore.getState();
      expect(past).toHaveLength(3);
      expect(past[1].nodes).toEqual([node1]);
      expect(past[2].nodes).toEqual([node2]);
    });
  });

  describe("edges in undo/redo", () => {
    it("restores edges on undo and redo", async () => {
      const edge = createTestEdge("e1", "1", "2");
      useEdgeStore.setState({ edges: [edge] });

      useHistoryStore.getState().pushState({ nodes: [], edges: [] });
      await flushMicrotasks();

      useHistoryStore.getState().undo();
      expect(useEdgeStore.getState().edges).toEqual([]);

      useHistoryStore.getState().redo();
      expect(useEdgeStore.getState().edges).toEqual([edge]);
    });
  });

  describe("pushState clears future", () => {
    it("clears future when a new state is pushed after undo", async () => {
      const node1 = createTestNode("1");
      const node2 = createTestNode("2");
      const node3 = createTestNode("3");

      // Initialize empty
      useHistoryStore.getState().initializeHistory();

      // First change: set [node1]
      useNodeStore.setState({ nodes: [node1] });

      // Second change: set [node1, node2], push pre-change [node1]
      useNodeStore.setState({ nodes: [node1, node2] });
      useHistoryStore.getState().pushState({ nodes: [node1], edges: [] });
      await flushMicrotasks();

      // Undo: back to [node1]
      useHistoryStore.getState().undo();
      expect(useHistoryStore.getState().future).toHaveLength(1);

      // New diverging change: add node3 instead of node2
      useNodeStore.setState({ nodes: [node1, node3] });
      useHistoryStore.getState().pushState({ nodes: [node1], edges: [] });
      await flushMicrotasks();

      expect(useHistoryStore.getState().future).toHaveLength(0);
    });
  });
});
