/**
 * Unit tests for the action applicator helpers.
 *
 * These cover graph-mutation logic (apply node input, connect nodes, undo
 * snapshots, clone helpers) in isolation from the hook that composes them,
 * so validation errors, idempotent no-ops, and the `structuredClone`
 * fallback path all have direct coverage.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { MarkerType } from "@xyflow/react";
import type { Dispatch, SetStateAction } from "react";
import type { CustomNode } from "../../FlowEditor/nodes/CustomNode/CustomNode";
import type { CustomEdge } from "../../FlowEditor/edges/CustomEdge";

// --- Module mocks ---

let mockNodes: CustomNode[] = [];
let mockEdges: CustomEdge[] = [];
const mockSetNodes = vi.fn();
const mockSetEdges = vi.fn();

vi.mock("../../../stores/nodeStore", () => {
  const useNodeStore = () => ({});
  useNodeStore.getState = () => ({
    nodes: mockNodes,
    setNodes: mockSetNodes,
  });
  return { useNodeStore };
});

vi.mock("../../../stores/edgeStore", () => {
  const useEdgeStore = () => ({});
  useEdgeStore.getState = () => ({
    edges: mockEdges,
    setEdges: mockSetEdges,
  });
  return { useEdgeStore };
});

// Import after mocks
import {
  DEFAULT_EDGE_MARKER_COLOR,
  MAX_UNDO,
  type ApplyActionDeps,
  type UndoSnapshot,
  applyConnectNodes,
  applyUpdateNodeInput,
  cloneNodes,
  pushUndoEntry,
  safeCloneArray,
} from "../actionApplicators";

// --- Test helpers ---

function makeNode(overrides: Partial<CustomNode> = {}): CustomNode {
  return {
    id: "node-1",
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      id: "node-1",
      title: "Test Node",
      inputSchema: {
        type: "object",
        properties: {
          text: { type: "string" },
          count: { type: "number" },
        },
      },
      outputSchema: {
        type: "object",
        properties: {
          result: { type: "string" },
        },
      },
      hardcodedValues: {},
      ...((overrides.data as object) ?? {}),
    },
    ...overrides,
  } as unknown as CustomNode;
}

interface TestDeps {
  toast: ReturnType<typeof vi.fn>;
  setNodes: typeof mockSetNodes;
  setEdges: typeof mockSetEdges;
  setUndoStack: ReturnType<typeof vi.fn>;
  setAppliedActionKeys: ReturnType<typeof vi.fn>;
}

function makeDeps(): TestDeps & ApplyActionDeps {
  const deps = {
    toast: vi.fn(),
    setNodes: mockSetNodes,
    setEdges: mockSetEdges,
    setUndoStack: vi.fn(),
    setAppliedActionKeys: vi.fn(),
  };
  // Cast through unknown — vi.fn mocks are structurally compatible with the
  // dispatch/toast signatures we use at runtime, but TypeScript's narrow type
  // definitions don't align directly.
  return deps as unknown as TestDeps & ApplyActionDeps;
}

beforeEach(() => {
  mockNodes = [];
  mockEdges = [];
  mockSetNodes.mockClear();
  mockSetEdges.mockClear();
});

// -----------------------------------------------------------------------
// safeCloneArray
// -----------------------------------------------------------------------

describe("safeCloneArray", () => {
  it("returns a deep clone when structuredClone is available", () => {
    const items = [{ a: 1, nested: { b: 2 } }];
    const cloned = safeCloneArray(items);
    expect(cloned).toEqual(items);
    expect(cloned).not.toBe(items);
    expect(cloned[0]).not.toBe(items[0]);
    // Mutating the clone must not affect the original.
    (cloned[0].nested as { b: number }).b = 999;
    expect(items[0].nested.b).toBe(2);
  });

  it("falls back to a shallow spread when structuredClone throws", () => {
    const original = globalThis.structuredClone;
    // Force the fallback path by making structuredClone throw.
    (globalThis as { structuredClone: unknown }).structuredClone = () => {
      throw new Error("not cloneable");
    };
    try {
      const items = [{ a: 1 }, { a: 2 }];
      const cloned = safeCloneArray(items);
      expect(cloned).toHaveLength(2);
      expect(cloned[0]).not.toBe(items[0]);
      expect(cloned[0].a).toBe(1);
    } finally {
      (globalThis as { structuredClone: unknown }).structuredClone = original;
    }
  });

  it("falls back when structuredClone is undefined", () => {
    const original = globalThis.structuredClone;
    (globalThis as { structuredClone: unknown }).structuredClone =
      undefined as unknown;
    try {
      const items = [{ x: 1 }];
      const cloned = safeCloneArray(items);
      expect(cloned).toEqual(items);
      expect(cloned[0]).not.toBe(items[0]);
    } finally {
      (globalThis as { structuredClone: unknown }).structuredClone = original;
    }
  });
});

// -----------------------------------------------------------------------
// cloneNodes
// -----------------------------------------------------------------------

describe("cloneNodes", () => {
  it("deep clones nodes via structuredClone", () => {
    const nodes = [makeNode({ id: "a" }), makeNode({ id: "b" })];
    const cloned = cloneNodes(nodes);
    expect(cloned).toHaveLength(2);
    expect(cloned[0]).not.toBe(nodes[0]);
    expect(cloned[0].data).not.toBe(nodes[0].data);
  });

  it("falls back to a shallow node+data copy when structuredClone fails", () => {
    const original = globalThis.structuredClone;
    (globalThis as { structuredClone: unknown }).structuredClone = () => {
      throw new Error("boom");
    };
    try {
      const nodes = [makeNode({ id: "a" })];
      const cloned = cloneNodes(nodes);
      expect(cloned[0]).not.toBe(nodes[0]);
      expect(cloned[0].data).not.toBe(nodes[0].data);
      // data still carries the original field values.
      expect(cloned[0].id).toBe("a");
    } finally {
      (globalThis as { structuredClone: unknown }).structuredClone = original;
    }
  });
});

// -----------------------------------------------------------------------
// pushUndoEntry
// -----------------------------------------------------------------------

describe("pushUndoEntry", () => {
  it("appends a new entry", () => {
    let stack: UndoSnapshot[] = [];
    const setter: Dispatch<SetStateAction<UndoSnapshot[]>> = (v) => {
      stack = typeof v === "function" ? v(stack) : v;
    };
    pushUndoEntry(setter, { actionKey: "k1", restore: () => {} });
    expect(stack).toHaveLength(1);
    expect(stack[0].actionKey).toBe("k1");
  });

  it("trims the oldest entry when MAX_UNDO is reached", () => {
    let stack: UndoSnapshot[] = Array.from({ length: MAX_UNDO }, (_, i) => ({
      actionKey: `k${i}`,
      restore: () => {},
    }));
    const setter: Dispatch<SetStateAction<UndoSnapshot[]>> = (v) => {
      stack = typeof v === "function" ? v(stack) : v;
    };
    pushUndoEntry(setter, { actionKey: "newest", restore: () => {} });
    expect(stack).toHaveLength(MAX_UNDO);
    // Oldest was dropped, newest is at the end.
    expect(stack[0].actionKey).toBe("k1");
    expect(stack[stack.length - 1].actionKey).toBe("newest");
  });
});

// -----------------------------------------------------------------------
// applyUpdateNodeInput
// -----------------------------------------------------------------------

describe("applyUpdateNodeInput", () => {
  it("rejects an action targeting a missing node", () => {
    mockNodes = [makeNode({ id: "node-1" })];
    const deps = makeDeps();
    const result = applyUpdateNodeInput(
      {
        type: "update_node_input",
        nodeId: "missing",
        key: "text",
        value: "v",
      },
      deps,
    );
    expect(result).toBe(false);
    expect(deps.toast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
    expect(mockSetNodes).not.toHaveBeenCalled();
  });

  it("blocks __proto__ as a prototype pollution guard", () => {
    mockNodes = [makeNode({ id: "node-1" })];
    const deps = makeDeps();
    const result = applyUpdateNodeInput(
      {
        type: "update_node_input",
        nodeId: "node-1",
        key: "__proto__",
        value: "polluted",
      },
      deps,
    );
    expect(result).toBe(false);
    expect(mockSetNodes).not.toHaveBeenCalled();
    // Prototype must be clean.
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });

  it("blocks constructor and prototype as dangerous keys", () => {
    mockNodes = [makeNode({ id: "node-1" })];
    const deps = makeDeps();
    for (const key of ["constructor", "prototype"]) {
      const result = applyUpdateNodeInput(
        { type: "update_node_input", nodeId: "node-1", key, value: "v" },
        deps,
      );
      expect(result).toBe(false);
    }
    expect(mockSetNodes).not.toHaveBeenCalled();
  });

  it("rejects keys not present in the node's input schema", () => {
    mockNodes = [makeNode({ id: "node-1" })];
    const deps = makeDeps();
    const result = applyUpdateNodeInput(
      {
        type: "update_node_input",
        nodeId: "node-1",
        key: "unknown_field",
        value: "v",
      },
      deps,
    );
    expect(result).toBe(false);
    expect(mockSetNodes).not.toHaveBeenCalled();
  });

  it("allows any key when the node has no input schema", () => {
    mockNodes = [
      makeNode({
        id: "node-1",
        data: {
          id: "node-1",
          title: "Schemaless",
          inputSchema: undefined,
          hardcodedValues: {},
        },
      } as unknown as CustomNode),
    ];
    const deps = makeDeps();
    const result = applyUpdateNodeInput(
      {
        type: "update_node_input",
        nodeId: "node-1",
        key: "anything",
        value: 42,
      },
      deps,
    );
    expect(result).toBe(true);
    expect(mockSetNodes).toHaveBeenCalledTimes(1);
  });

  it("applies a valid update and pushes an undo snapshot", () => {
    mockNodes = [
      makeNode({
        id: "node-1",
        data: {
          id: "node-1",
          title: "T",
          inputSchema: { type: "object", properties: { text: {} } },
          hardcodedValues: { text: "old" },
        },
      } as unknown as CustomNode),
    ];
    const deps = makeDeps();
    const result = applyUpdateNodeInput(
      {
        type: "update_node_input",
        nodeId: "node-1",
        key: "text",
        value: "new",
      },
      deps,
    );
    expect(result).toBe(true);
    expect(mockSetNodes).toHaveBeenCalledTimes(1);
    const nextNodes = mockSetNodes.mock.calls[0][0];
    expect(nextNodes[0].data.hardcodedValues.text).toBe("new");
    expect(deps.setUndoStack).toHaveBeenCalledTimes(1);
  });

  it("undo reverts only the target field and preserves later edits to other fields", () => {
    const original = {
      id: "node-1",
      title: "T",
      inputSchema: {
        type: "object",
        properties: { text: {}, other: {} },
      },
      hardcodedValues: { text: "old", other: "untouched" },
    };
    mockNodes = [
      makeNode({ id: "node-1", data: original } as unknown as CustomNode),
    ];
    const deps = makeDeps();
    applyUpdateNodeInput(
      {
        type: "update_node_input",
        nodeId: "node-1",
        key: "text",
        value: "new",
      },
      deps,
    );
    // Extract the snapshot's restore closure via the setUndoStack mock.
    const updater = deps.setUndoStack.mock.calls[0][0];
    const stack = updater([]);
    expect(stack).toHaveLength(1);
    const entry = stack[0];

    // Simulate a later edit to an unrelated field on the same node by
    // replacing the live node with an updated version — this mirrors what
    // setNodes(…) does in production.
    mockNodes = [
      makeNode({
        id: "node-1",
        data: {
          id: "node-1",
          title: "T",
          inputSchema: original.inputSchema,
          hardcodedValues: { text: "new", other: "edited-after-apply" },
        },
      } as unknown as CustomNode),
    ];

    mockSetNodes.mockClear();
    entry.restore();
    expect(mockSetNodes).toHaveBeenCalledTimes(1);
    const restoredNodes = mockSetNodes.mock.calls[0][0];
    const hardcoded = (
      restoredNodes[0].data as { hardcodedValues: Record<string, string> }
    ).hardcodedValues;
    // `text` should be reverted to the pre-apply value.
    expect(hardcoded.text).toBe("old");
    // The later unrelated edit must be preserved (differential undo).
    expect(hardcoded.other).toBe("edited-after-apply");
  });

  it("undo removes a newly-added key when the field did not exist pre-apply", () => {
    const original = {
      id: "node-1",
      title: "T",
      inputSchema: { type: "object", properties: { text: {} } },
      hardcodedValues: {},
    };
    mockNodes = [
      makeNode({ id: "node-1", data: original } as unknown as CustomNode),
    ];
    const deps = makeDeps();
    applyUpdateNodeInput(
      {
        type: "update_node_input",
        nodeId: "node-1",
        key: "text",
        value: "new",
      },
      deps,
    );
    const stack = deps.setUndoStack.mock.calls[0][0]([]);
    mockSetNodes.mockClear();
    stack[0].restore();
    const restoredNodes = mockSetNodes.mock.calls[0][0];
    const hardcoded = (
      restoredNodes[0].data as { hardcodedValues: Record<string, unknown> }
    ).hardcodedValues;
    // Key did not exist before apply → undo should remove it entirely.
    expect(Object.prototype.hasOwnProperty.call(hardcoded, "text")).toBe(false);
  });
});

// -----------------------------------------------------------------------
// applyConnectNodes
// -----------------------------------------------------------------------

describe("applyConnectNodes", () => {
  beforeEach(() => {
    mockNodes = [
      makeNode({
        id: "src",
        data: {
          id: "src",
          title: "Source",
          outputSchema: { type: "object", properties: { result: {} } },
          hardcodedValues: {},
        },
      } as unknown as CustomNode),
      makeNode({
        id: "dst",
        data: {
          id: "dst",
          title: "Dest",
          inputSchema: { type: "object", properties: { text: {} } },
          hardcodedValues: {},
        },
      } as unknown as CustomNode),
    ];
    mockEdges = [];
  });

  it("rejects a connection when source node is missing", () => {
    const deps = makeDeps();
    const result = applyConnectNodes(
      {
        type: "connect_nodes",
        source: "missing",
        sourceHandle: "result",
        target: "dst",
        targetHandle: "text",
      },
      deps,
    );
    expect(result).toBe(false);
    expect(mockSetEdges).not.toHaveBeenCalled();
  });

  it("rejects a connection when target node is missing", () => {
    const deps = makeDeps();
    const result = applyConnectNodes(
      {
        type: "connect_nodes",
        source: "src",
        sourceHandle: "result",
        target: "missing",
        targetHandle: "text",
      },
      deps,
    );
    expect(result).toBe(false);
    expect(mockSetEdges).not.toHaveBeenCalled();
  });

  it("rejects a connection when source handle is not in outputSchema", () => {
    const deps = makeDeps();
    const result = applyConnectNodes(
      {
        type: "connect_nodes",
        source: "src",
        sourceHandle: "nope",
        target: "dst",
        targetHandle: "text",
      },
      deps,
    );
    expect(result).toBe(false);
    expect(mockSetEdges).not.toHaveBeenCalled();
  });

  it("rejects a connection when target handle is not in inputSchema", () => {
    const deps = makeDeps();
    const result = applyConnectNodes(
      {
        type: "connect_nodes",
        source: "src",
        sourceHandle: "result",
        target: "dst",
        targetHandle: "nope",
      },
      deps,
    );
    expect(result).toBe(false);
    expect(mockSetEdges).not.toHaveBeenCalled();
  });

  it("creates a new edge with the default marker color on success", () => {
    const deps = makeDeps();
    const result = applyConnectNodes(
      {
        type: "connect_nodes",
        source: "src",
        sourceHandle: "result",
        target: "dst",
        targetHandle: "text",
      },
      deps,
    );
    expect(result).toBe(true);
    expect(mockSetEdges).toHaveBeenCalledTimes(1);
    const newEdges = mockSetEdges.mock.calls[0][0];
    expect(newEdges).toHaveLength(1);
    expect(newEdges[0]).toMatchObject({
      source: "src",
      target: "dst",
      sourceHandle: "result",
      targetHandle: "text",
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: DEFAULT_EDGE_MARKER_COLOR,
      },
    });
    expect(deps.setUndoStack).toHaveBeenCalledTimes(1);
  });

  it("is idempotent when the same edge already exists", () => {
    mockEdges = [
      {
        id: "existing",
        source: "src",
        target: "dst",
        sourceHandle: "result",
        targetHandle: "text",
        type: "custom",
      } as unknown as CustomEdge,
    ];
    const deps = makeDeps();
    const result = applyConnectNodes(
      {
        type: "connect_nodes",
        source: "src",
        sourceHandle: "result",
        target: "dst",
        targetHandle: "text",
      },
      deps,
    );
    expect(result).toBe(true);
    // No new edge written; applied key still marked.
    expect(mockSetEdges).not.toHaveBeenCalled();
    expect(deps.setAppliedActionKeys).toHaveBeenCalledTimes(1);
    // No undo entry for a no-op.
    expect(deps.setUndoStack).not.toHaveBeenCalled();
  });

  it("undo removes only the AI-added edge and preserves later edits", () => {
    mockEdges = [
      {
        id: "other",
        source: "a",
        target: "b",
        sourceHandle: "x",
        targetHandle: "y",
        type: "custom",
      } as unknown as CustomEdge,
    ];
    const deps = makeDeps();
    applyConnectNodes(
      {
        type: "connect_nodes",
        source: "src",
        sourceHandle: "result",
        target: "dst",
        targetHandle: "text",
      },
      deps,
    );
    const stack = deps.setUndoStack.mock.calls[0][0]([]);
    expect(stack).toHaveLength(1);

    // Simulate a later user edit — the applied edge plus a brand new
    // manually-added edge. Differential undo should only drop the former.
    mockEdges = [
      {
        id: "other",
        source: "a",
        target: "b",
        sourceHandle: "x",
        targetHandle: "y",
        type: "custom",
      } as unknown as CustomEdge,
      {
        id: "src:result->dst:text",
        source: "src",
        target: "dst",
        sourceHandle: "result",
        targetHandle: "text",
        type: "custom",
      } as unknown as CustomEdge,
      {
        id: "later-manual-edge",
        source: "a",
        target: "dst",
        sourceHandle: "x",
        targetHandle: "text",
        type: "custom",
      } as unknown as CustomEdge,
    ];

    mockSetEdges.mockClear();
    stack[0].restore();
    expect(mockSetEdges).toHaveBeenCalledTimes(1);
    const restored = mockSetEdges.mock.calls[0][0];
    // Should contain the pre-existing edge AND the later manual edge.
    // Only the AI-applied edge should be removed.
    expect(restored).toHaveLength(2);
    const ids = restored.map((e: CustomEdge) => e.id).sort();
    expect(ids).toEqual(["later-manual-edge", "other"]);
  });
});
