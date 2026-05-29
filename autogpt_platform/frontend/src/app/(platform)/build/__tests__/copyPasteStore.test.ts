import { describe, it, expect, beforeEach, vi } from "vitest";
import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import { BlockUIType } from "../components/types";

vi.mock("@/services/storage/local-storage", () => {
  const store: Record<string, string> = {};
  return {
    Key: { COPIED_FLOW_DATA: "COPIED_FLOW_DATA" },
    storage: {
      get: (key: string) => store[key] ?? null,
      set: (key: string, value: string) => {
        store[key] = value;
      },
      clean: (key: string) => {
        delete store[key];
      },
    },
  };
});

import { useCopyPasteStore } from "../stores/copyPasteStore";
import { useNodeStore } from "../stores/nodeStore";
import { useEdgeStore } from "../stores/edgeStore";
import { useHistoryStore } from "../stores/historyStore";
import { storage, Key } from "@/services/storage/local-storage";

function createTestNode(
  id: string,
  overrides: Partial<CustomNode> = {},
): CustomNode {
  return {
    id,
    type: "custom",
    position: overrides.position ?? { x: 100, y: 200 },
    selected: overrides.selected,
    data: {
      hardcodedValues: {},
      title: `Node ${id}`,
      description: "test node",
      inputSchema: {},
      outputSchema: {},
      uiType: BlockUIType.STANDARD,
      block_id: `block-${id}`,
      costs: [],
      categories: [],
      ...overrides.data,
    },
  } as CustomNode;
}

describe("useCopyPasteStore", () => {
  beforeEach(() => {
    useNodeStore.setState({ nodes: [], nodeCounter: 0 });
    useEdgeStore.setState({ edges: [] });
    useHistoryStore.getState().clear();
    storage.clean(Key.COPIED_FLOW_DATA);
  });

  describe("copySelectedNodes", () => {
    it("copies a single selected node to localStorage", () => {
      const node = createTestNode("1", { selected: true });
      useNodeStore.setState({ nodes: [node] });

      useCopyPasteStore.getState().copySelectedNodes();

      const stored = storage.get(Key.COPIED_FLOW_DATA);
      expect(stored).not.toBeNull();

      const parsed = JSON.parse(stored!);
      expect(parsed.nodes).toHaveLength(1);
      expect(parsed.nodes[0].id).toBe("1");
      expect(parsed.edges).toHaveLength(0);
    });

    it("copies only edges between selected nodes", () => {
      const nodeA = createTestNode("a", { selected: true });
      const nodeB = createTestNode("b", { selected: true });
      const nodeC = createTestNode("c", { selected: false });
      useNodeStore.setState({ nodes: [nodeA, nodeB, nodeC] });

      useEdgeStore.setState({
        edges: [
          {
            id: "e-ab",
            source: "a",
            target: "b",
            sourceHandle: "out",
            targetHandle: "in",
          },
          {
            id: "e-bc",
            source: "b",
            target: "c",
            sourceHandle: "out",
            targetHandle: "in",
          },
          {
            id: "e-ac",
            source: "a",
            target: "c",
            sourceHandle: "out",
            targetHandle: "in",
          },
        ],
      });

      useCopyPasteStore.getState().copySelectedNodes();

      const parsed = JSON.parse(storage.get(Key.COPIED_FLOW_DATA)!);
      expect(parsed.nodes).toHaveLength(2);
      expect(parsed.edges).toHaveLength(1);
      expect(parsed.edges[0].id).toBe("e-ab");
    });

    it("stores empty data when no nodes are selected", () => {
      const node = createTestNode("1", { selected: false });
      useNodeStore.setState({ nodes: [node] });

      useCopyPasteStore.getState().copySelectedNodes();

      const parsed = JSON.parse(storage.get(Key.COPIED_FLOW_DATA)!);
      expect(parsed.nodes).toHaveLength(0);
      expect(parsed.edges).toHaveLength(0);
    });
  });

  describe("pasteNodes", () => {
    it("creates new nodes with new IDs via incrementNodeCounter", () => {
      const node = createTestNode("orig", {
        selected: true,
        position: { x: 100, y: 200 },
      });
      useNodeStore.setState({ nodes: [node], nodeCounter: 5 });

      useCopyPasteStore.getState().copySelectedNodes();
      useCopyPasteStore.getState().pasteNodes();

      const { nodes } = useNodeStore.getState();
      expect(nodes).toHaveLength(2);

      const pastedNode = nodes.find((n) => n.id !== "orig");
      expect(pastedNode).toBeDefined();
      expect(pastedNode!.id).not.toBe("orig");
    });

    it("offsets pasted node positions by +50 x/y", () => {
      const node = createTestNode("orig", {
        selected: true,
        position: { x: 100, y: 200 },
      });
      useNodeStore.setState({ nodes: [node], nodeCounter: 5 });

      useCopyPasteStore.getState().copySelectedNodes();
      useCopyPasteStore.getState().pasteNodes();

      const { nodes } = useNodeStore.getState();
      const pastedNode = nodes.find((n) => n.id !== "orig");
      expect(pastedNode).toBeDefined();
      expect(pastedNode!.position).toEqual({ x: 150, y: 250 });
    });

    it("preserves internal connections with remapped IDs", () => {
      const nodeA = createTestNode("a", {
        selected: true,
        position: { x: 0, y: 0 },
      });
      const nodeB = createTestNode("b", {
        selected: true,
        position: { x: 200, y: 0 },
      });
      useNodeStore.setState({ nodes: [nodeA, nodeB], nodeCounter: 0 });
      useEdgeStore.setState({
        edges: [
          {
            id: "e-ab",
            source: "a",
            target: "b",
            sourceHandle: "output",
            targetHandle: "input",
          },
        ],
      });

      useCopyPasteStore.getState().copySelectedNodes();
      useCopyPasteStore.getState().pasteNodes();

      const { edges } = useEdgeStore.getState();
      const newEdges = edges.filter((e) => e.id !== "e-ab");
      expect(newEdges).toHaveLength(1);

      const newEdge = newEdges[0];
      expect(newEdge.source).not.toBe("a");
      expect(newEdge.target).not.toBe("b");

      const { nodes } = useNodeStore.getState();
      const pastedNodeIDs = nodes
        .filter((n) => n.id !== "a" && n.id !== "b")
        .map((n) => n.id);

      expect(pastedNodeIDs).toContain(newEdge.source);
      expect(pastedNodeIDs).toContain(newEdge.target);
    });

    it("deselects existing nodes and selects pasted ones", () => {
      const existingNode = createTestNode("existing", {
        selected: true,
        position: { x: 0, y: 0 },
      });
      const nodeToCopy = createTestNode("copy-me", {
        selected: true,
        position: { x: 100, y: 100 },
      });
      useNodeStore.setState({
        nodes: [existingNode, nodeToCopy],
        nodeCounter: 0,
      });

      useCopyPasteStore.getState().copySelectedNodes();

      // Deselect nodeToCopy, keep existingNode selected to verify deselection on paste
      useNodeStore.setState({
        nodes: [
          { ...existingNode, selected: true },
          { ...nodeToCopy, selected: false },
        ],
      });

      useCopyPasteStore.getState().pasteNodes();

      const { nodes } = useNodeStore.getState();
      const originalNodes = nodes.filter(
        (n) => n.id === "existing" || n.id === "copy-me",
      );
      const pastedNodes = nodes.filter(
        (n) => n.id !== "existing" && n.id !== "copy-me",
      );

      originalNodes.forEach((n) => {
        expect(n.selected).toBe(false);
      });
      pastedNodes.forEach((n) => {
        expect(n.selected).toBe(true);
      });
    });

    it("does nothing when clipboard is empty", () => {
      const node = createTestNode("1", { position: { x: 0, y: 0 } });
      useNodeStore.setState({ nodes: [node], nodeCounter: 0 });

      useCopyPasteStore.getState().pasteNodes();

      const { nodes } = useNodeStore.getState();
      expect(nodes).toHaveLength(1);
      expect(nodes[0].id).toBe("1");
    });
  });
});
