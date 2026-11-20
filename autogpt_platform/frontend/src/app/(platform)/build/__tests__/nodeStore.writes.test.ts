import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useNodeStore } from "../stores/nodeStore";
import { useHistoryStore } from "../stores/historyStore";
import { useEdgeStore } from "../stores/edgeStore";
import { BlockUIType } from "../components/types";
import type {
  CustomNode,
  CustomNodeData,
} from "../components/FlowEditor/nodes/CustomNode/CustomNode";

function createTestNode(id: string, data: Partial<CustomNodeData> = {}) {
  const node: CustomNode = {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues: {},
      title: `Node ${id}`,
      description: "",
      inputSchema: {},
      outputSchema: {},
      uiType: BlockUIType.STANDARD,
      block_id: "test-block-id",
      costs: [],
      categories: [],
      ...data,
    },
  };
  return node;
}

let nodeWrites = 0;
let unsubscribe: () => void;

beforeEach(() => {
  useNodeStore.setState({ nodes: [] });
  useEdgeStore.setState({ edges: [] });
  useHistoryStore.getState().clear();
  nodeWrites = 0;
  unsubscribe = useNodeStore.subscribe((state, prev) => {
    if (state.nodes !== prev.nodes) nodeWrites++;
  });
});

afterEach(() => {
  unsubscribe();
});

async function flushHistory() {
  await Promise.resolve();
}

describe("addNodes", () => {
  it("adds a whole workflow in one store write", () => {
    const nodes = Array.from({ length: 100 }, (_, i) =>
      createTestNode(String(i)),
    );

    useNodeStore.getState().addNodes(nodes);

    expect(nodeWrites).toBe(1);
    expect(useNodeStore.getState().nodes).toEqual(nodes);
  });

  it("appends after the nodes already on the canvas, in order", () => {
    const existing = createTestNode("a");
    useNodeStore.getState().setNodes([existing]);
    nodeWrites = 0;

    useNodeStore
      .getState()
      .addNodes([createTestNode("b"), createTestNode("c")]);

    expect(nodeWrites).toBe(1);
    expect(useNodeStore.getState().nodes.map((n) => n.id)).toEqual([
      "a",
      "b",
      "c",
    ]);
    expect(useNodeStore.getState().nodes[0]).toBe(existing);
  });

  it("writes nothing for an empty list", () => {
    useNodeStore.getState().addNodes([]);

    expect(nodeWrites).toBe(0);
  });

  it("does not add an undo entry, as before", async () => {
    useNodeStore.getState().addNodes([createTestNode("1")]);
    await flushHistory();

    expect(useHistoryStore.getState().past).toHaveLength(1);
  });

  it("finds the added nodes by id", () => {
    useNodeStore
      .getState()
      .addNodes([
        createTestNode("1", { hardcodedValues: { a: 1 } }),
        createTestNode("2", { hardcodedValues: { a: 2 } }),
      ]);

    expect(useNodeStore.getState().getHardCodedValues("2")).toEqual({ a: 2 });
  });
});

describe("updateNodeData", () => {
  it("a changed value writes once and can be undone", async () => {
    useNodeStore.getState().setNodes([createTestNode("1")]);
    useHistoryStore.getState().initializeHistory();
    nodeWrites = 0;

    useNodeStore
      .getState()
      .updateNodeData("1", { hardcodedValues: { text: "hello" } });
    await flushHistory();

    expect(nodeWrites).toBe(1);
    expect(useNodeStore.getState().nodes[0].data.hardcodedValues).toEqual({
      text: "hello",
    });
    expect(useHistoryStore.getState().canUndo()).toBe(true);

    useHistoryStore.getState().undo();
    expect(useNodeStore.getState().nodes[0].data.hardcodedValues).toEqual({});
  });

  it("an equal value writes nothing and keeps the node object", async () => {
    const node = createTestNode("1", { hardcodedValues: { text: "same" } });
    useNodeStore.getState().setNodes([node]);
    useHistoryStore.getState().initializeHistory();
    nodeWrites = 0;

    useNodeStore
      .getState()
      .updateNodeData("1", { hardcodedValues: { text: "same" } });
    await flushHistory();

    expect(nodeWrites).toBe(0);
    expect(useNodeStore.getState().nodes[0]).toBe(node);
    expect(useHistoryStore.getState().canUndo()).toBe(false);
  });

  it("an unknown node writes nothing", () => {
    useNodeStore.getState().setNodes([createTestNode("1")]);
    nodeWrites = 0;

    useNodeStore.getState().updateNodeData("missing", { title: "x" });

    expect(nodeWrites).toBe(0);
  });

  it("a key the node lacks is still written when its value is undefined", () => {
    useNodeStore.getState().setNodes([createTestNode("1")]);
    nodeWrites = 0;

    useNodeStore.getState().updateNodeData("1", { status: undefined });

    expect(nodeWrites).toBe(1);
    expect(Object.hasOwn(useNodeStore.getState().nodes[0].data, "status")).toBe(
      true,
    );
  });

  it("one changed key among equal ones still writes", () => {
    useNodeStore
      .getState()
      .setNodes([createTestNode("1", { title: "Same", hardcodedValues: {} })]);
    nodeWrites = 0;

    useNodeStore
      .getState()
      .updateNodeData("1", { title: "Same", hardcodedValues: { a: 1 } });

    expect(nodeWrites).toBe(1);
    expect(useNodeStore.getState().nodes[0].data.hardcodedValues).toEqual({
      a: 1,
    });
  });

  it("updates every node that shares the id, as before", () => {
    useNodeStore
      .getState()
      .setNodes([
        createTestNode("1", { title: "Changed" }),
        createTestNode("1", { title: "Old" }),
      ]);

    useNodeStore.getState().updateNodeData("1", { title: "Changed" });

    expect(useNodeStore.getState().nodes.map((n) => n.data.title)).toEqual([
      "Changed",
      "Changed",
    ]);
  });
});
