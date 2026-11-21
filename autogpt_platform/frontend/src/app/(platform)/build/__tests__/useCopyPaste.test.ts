import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { CustomNode } from "../components/FlowEditor/nodes/CustomNode/CustomNode";
import { BlockUIType } from "../components/types";

// ---- Mocks ----

const mockGetViewport = vi.fn(() => ({ x: 0, y: 0, zoom: 1 }));

vi.mock("@xyflow/react", async () => {
  const actual = await vi.importActual("@xyflow/react");
  return {
    ...actual,
    useReactFlow: vi.fn(() => ({
      getViewport: mockGetViewport,
    })),
  };
});

const mockToast = vi.fn();

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: vi.fn(() => ({ toast: mockToast })),
}));

let uuidCounter = 0;
vi.mock("uuid", () => ({
  v4: vi.fn(() => `new-uuid-${++uuidCounter}`),
}));

// Mock navigator.clipboard
const mockWriteText = vi.fn(() => Promise.resolve());
const mockReadText = vi.fn(() => Promise.resolve(""));

Object.defineProperty(navigator, "clipboard", {
  value: {
    writeText: mockWriteText,
    readText: mockReadText,
  },
  writable: true,
  configurable: true,
});

// Mock window.innerWidth / innerHeight for viewport centering calculations
Object.defineProperty(window, "innerWidth", { value: 1000, writable: true });
Object.defineProperty(window, "innerHeight", { value: 800, writable: true });

import { useCopyPaste } from "../components/FlowEditor/Flow/useCopyPaste";
import { useNodeStore } from "../stores/nodeStore";
import { useEdgeStore } from "../stores/edgeStore";
import { useHistoryStore } from "../stores/historyStore";
import { CustomEdge } from "../components/FlowEditor/edges/CustomEdge";

const CLIPBOARD_PREFIX = "autogpt-flow-data:";

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

function createTestEdge(
  id: string,
  source: string,
  target: string,
  sourceHandle = "out",
  targetHandle = "in",
): CustomEdge {
  return {
    id,
    source,
    target,
    sourceHandle,
    targetHandle,
  } as CustomEdge;
}

describe("useCopyPaste", () => {
  beforeEach(() => {
    useNodeStore.setState({ nodes: [], nodeCounter: 0 });
    useEdgeStore.setState({ edges: [] });
    useHistoryStore.getState().clear();
    mockWriteText.mockClear();
    mockReadText.mockClear();
    mockToast.mockClear();
    mockGetViewport.mockReturnValue({ x: 0, y: 0, zoom: 1 });
    uuidCounter = 0;

    // Ensure no input element is focused
    if (document.activeElement && document.activeElement !== document.body) {
      (document.activeElement as HTMLElement).blur();
    }
  });

  describe("copy (Ctrl+C)", () => {
    it("copies a single selected node to clipboard with prefix", async () => {
      const node = createTestNode("1", { selected: true });
      useNodeStore.setState({ nodes: [node] });

      press(keydown("c"));

      const written = await writtenClipboard();
      expect(written.startsWith(CLIPBOARD_PREFIX)).toBe(true);

      const parsed = parsePayload(written);
      expect(parsed.nodes).toHaveLength(1);
      expect(parsed.nodes[0].id).toBe("1");
      expect(parsed.edges).toHaveLength(0);
    });

    it("shows a success toast after copying", async () => {
      const node = createTestNode("1", { selected: true });
      useNodeStore.setState({ nodes: [node] });

      press(keydown("c"));

      await vi.waitFor(() => {
        expect(mockToast).toHaveBeenCalledWith(
          expect.objectContaining({
            title: "Copied successfully",
          }),
        );
      });
    });

    it("copies multiple connected nodes and preserves internal edges", async () => {
      const nodeA = createTestNode("a", { selected: true });
      const nodeB = createTestNode("b", { selected: true });
      const nodeC = createTestNode("c", { selected: false });
      useNodeStore.setState({ nodes: [nodeA, nodeB, nodeC] });

      useEdgeStore.setState({
        edges: [
          createTestEdge("e-ab", "a", "b"),
          createTestEdge("e-bc", "b", "c"),
        ],
      });

      press(keydown("c"));

      const parsed = parsePayload(await writtenClipboard());
      expect(parsed.nodes).toHaveLength(2);
      expect(parsed.edges).toHaveLength(1);
      expect(parsed.edges[0].id).toBe("e-ab");
    });

    it("drops external edges where one endpoint is not selected", async () => {
      const nodeA = createTestNode("a", { selected: true });
      const nodeB = createTestNode("b", { selected: false });
      useNodeStore.setState({ nodes: [nodeA, nodeB] });

      useEdgeStore.setState({
        edges: [createTestEdge("e-ab", "a", "b")],
      });

      press(keydown("c"));

      const parsed = parsePayload(await writtenClipboard());
      expect(parsed.nodes).toHaveLength(1);
      expect(parsed.edges).toHaveLength(0);
    });

    it("copies nothing when no nodes are selected", async () => {
      const node = createTestNode("1", { selected: false });
      useNodeStore.setState({ nodes: [node] });

      press(keydown("c"));

      const parsed = parsePayload(await writtenClipboard());
      expect(parsed.nodes).toHaveLength(0);
      expect(parsed.edges).toHaveLength(0);
    });
  });

  describe("paste (Ctrl+V)", () => {
    it("creates new nodes with new UUIDs", async () => {
      const node = createTestNode("orig", {
        selected: true,
        position: { x: 100, y: 200 },
      });

      clipboardHolds([node]);

      press(keydown("v"));

      const nodes = await pastedNodes(1);
      expect(nodes[0].id).toBe("new-uuid-1");
      expect(nodes[0].id).not.toBe("orig");
    });

    it("centers pasted nodes in the current viewport", async () => {
      // Viewport at origin, zoom 1 => center = (500, 400)
      mockGetViewport.mockReturnValue({ x: 0, y: 0, zoom: 1 });

      const node = createTestNode("orig", {
        selected: true,
        position: { x: 100, y: 100 },
      });

      clipboardHolds([node]);

      press(keydown("v"));

      const nodes = await pastedNodes(1);
      // Single node: center of bounds = (100, 100)
      // Viewport center = (500, 400)
      // Offset = (400, 300)
      // New position = (100 + 400, 100 + 300) = (500, 400)
      expect(nodes[0].position).toEqual({ x: 500, y: 400 });
    });

    it("deselects existing nodes and selects pasted nodes", async () => {
      const existingNode = createTestNode("existing", {
        selected: true,
        position: { x: 0, y: 0 },
      });

      useNodeStore.setState({ nodes: [existingNode], nodeCounter: 0 });

      const nodeToPaste = createTestNode("paste-me", {
        selected: false,
        position: { x: 100, y: 100 },
      });

      clipboardHolds([nodeToPaste]);

      press(keydown("v"));

      const nodes = await pastedNodes(2);
      const originalNode = nodes.find((n) => n.id === "existing");
      const pastedNode = nodes.find((n) => n.id !== "existing");

      expect(originalNode!.selected).toBe(false);
      expect(pastedNode!.selected).toBe(true);
    });

    it("remaps edge source/target IDs to newly created node IDs", async () => {
      const nodeA = createTestNode("a", {
        selected: true,
        position: { x: 0, y: 0 },
      });
      const nodeB = createTestNode("b", {
        selected: true,
        position: { x: 200, y: 0 },
      });
      const edge = createTestEdge("e-ab", "a", "b", "output", "input");

      clipboardHolds([nodeA, nodeB], [edge]);

      press(keydown("v"));

      await pastedNodes(2);

      // Wait for edges to be added too
      await vi.waitFor(() => {
        const { edges } = useEdgeStore.getState();
        expect(edges).toHaveLength(1);
      });

      const { edges } = useEdgeStore.getState();
      const newEdge = edges[0];

      // Edge source/target should be remapped to new UUIDs, not "a"/"b"
      expect(newEdge.source).not.toBe("a");
      expect(newEdge.target).not.toBe("b");
      expect(newEdge.source).toBe("new-uuid-1");
      expect(newEdge.target).toBe("new-uuid-2");
      expect(newEdge.sourceHandle).toBe("output");
      expect(newEdge.targetHandle).toBe("input");
    });

    it.each([
      [
        "does nothing when clipboard does not have the expected prefix",
        "some random text",
      ],
      ["does nothing when clipboard is empty", ""],
    ])("%s", async (_title, clipboardText) => {
      mockReadText.mockResolvedValue(clipboardText);

      const existingNode = createTestNode("1", { position: { x: 0, y: 0 } });
      useNodeStore.setState({ nodes: [existingNode], nodeCounter: 0 });

      press(keydown("v"));

      await vi.waitFor(() => {
        expect(mockReadText).toHaveBeenCalled();
      });

      // Ensure no state changes happen after clipboard read
      await vi.waitFor(() => {
        const { nodes } = useNodeStore.getState();
        expect(nodes).toHaveLength(1);
        expect(nodes[0].id).toBe("1");
      });
    });
  });

  describe("input field focus guard", () => {
    it("ignores Ctrl+C when an input element is focused", async () => {
      const node = createTestNode("1", { selected: true });
      useNodeStore.setState({ nodes: [node] });

      const input = document.createElement("input");
      document.body.appendChild(input);
      input.focus();

      press(keydown("c"));

      // Clipboard write should NOT be called
      expect(mockWriteText).not.toHaveBeenCalled();

      document.body.removeChild(input);
    });

    it("ignores Ctrl+V when a textarea element is focused", async () => {
      clipboardHolds([createTestNode("a", { position: { x: 0, y: 0 } })]);

      const textarea = document.createElement("textarea");
      document.body.appendChild(textarea);
      textarea.focus();

      press(keydown("v"));

      expect(mockReadText).not.toHaveBeenCalled();

      const { nodes } = useNodeStore.getState();
      expect(nodes).toHaveLength(0);

      document.body.removeChild(textarea);
    });

    it("ignores keypresses when a contenteditable element is focused", async () => {
      const node = createTestNode("1", { selected: true });
      useNodeStore.setState({ nodes: [node] });

      const div = document.createElement("div");
      div.setAttribute("contenteditable", "true");
      document.body.appendChild(div);
      div.focus();

      press(keydown("c"));

      expect(mockWriteText).not.toHaveBeenCalled();

      document.body.removeChild(div);
    });
  });

  describe("read-only graph", () => {
    it("ignores Ctrl+C when the graph is read-only", async () => {
      const node = createTestNode("1", { selected: true });
      useNodeStore.setState({ nodes: [node] });

      press(keydown("c"), true);

      expect(mockWriteText).not.toHaveBeenCalled();
    });

    it("ignores Ctrl+V when the graph is read-only", async () => {
      clipboardHolds([createTestNode("a", { position: { x: 0, y: 0 } })]);

      press(keydown("v"), true);

      expect(mockReadText).not.toHaveBeenCalled();
      expect(useNodeStore.getState().nodes).toHaveLength(0);
    });
  });

  describe("meta key support (macOS)", () => {
    it("handles Cmd+C (metaKey) the same as Ctrl+C", async () => {
      const node = createTestNode("1", { selected: true });
      useNodeStore.setState({ nodes: [node] });

      press(keydown("c", "metaKey"));

      await writtenClipboard();
    });

    it("handles Cmd+V (metaKey) the same as Ctrl+V", async () => {
      const node = createTestNode("orig", {
        selected: true,
        position: { x: 0, y: 0 },
      });
      clipboardHolds([node]);

      press(keydown("v", "metaKey"));

      await pastedNodes(1);
    });
  });
});

// Arguments after the event go to the hook as given, so an omitted one takes its default.
function press(
  event: KeyboardEvent,
  ...hookArgs: Parameters<typeof useCopyPaste>
) {
  const { result } = renderHook(() => useCopyPaste(...hookArgs));
  act(() => {
    result.current(event);
  });
}

function keydown(
  key: "c" | "v",
  modifier: "ctrlKey" | "metaKey" = "ctrlKey",
): KeyboardEvent {
  return new KeyboardEvent("keydown", { key, [modifier]: true, bubbles: true });
}

function clipboardHolds(nodes: CustomNode[], edges: CustomEdge[] = []) {
  mockReadText.mockResolvedValue(
    `${CLIPBOARD_PREFIX}${JSON.stringify({ nodes, edges })}`,
  );
}

async function writtenClipboard(): Promise<string> {
  await vi.waitFor(() => {
    expect(mockWriteText).toHaveBeenCalledTimes(1);
  });
  return (mockWriteText.mock.calls as string[][])[0][0];
}

function parsePayload(written: string) {
  return JSON.parse(written.slice(CLIPBOARD_PREFIX.length));
}

async function pastedNodes(count: number): Promise<CustomNode[]> {
  await vi.waitFor(() => {
    const { nodes } = useNodeStore.getState();
    expect(nodes).toHaveLength(count);
  });
  return useNodeStore.getState().nodes;
}
