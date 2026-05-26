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

function makeCopyEvent(): KeyboardEvent {
  return new KeyboardEvent("keydown", {
    key: "c",
    ctrlKey: true,
    bubbles: true,
  });
}

function makePasteEvent(): KeyboardEvent {
  return new KeyboardEvent("keydown", {
    key: "v",
    ctrlKey: true,
    bubbles: true,
  });
}

function clipboardPayload(nodes: CustomNode[], edges: CustomEdge[]): string {
  return `${CLIPBOARD_PREFIX}${JSON.stringify({ nodes, edges })}`;
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

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makeCopyEvent());
      });

      await vi.waitFor(() => {
        expect(mockWriteText).toHaveBeenCalledTimes(1);
      });

      const written = (mockWriteText.mock.calls as string[][])[0][0];
      expect(written.startsWith(CLIPBOARD_PREFIX)).toBe(true);

      const parsed = JSON.parse(written.slice(CLIPBOARD_PREFIX.length));
      expect(parsed.nodes).toHaveLength(1);
      expect(parsed.nodes[0].id).toBe("1");
      expect(parsed.edges).toHaveLength(0);
    });

    it("shows a success toast after copying", async () => {
      const node = createTestNode("1", { selected: true });
      useNodeStore.setState({ nodes: [node] });

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makeCopyEvent());
      });

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

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makeCopyEvent());
      });

      await vi.waitFor(() => {
        expect(mockWriteText).toHaveBeenCalledTimes(1);
      });

      const parsed = JSON.parse(
        (mockWriteText.mock.calls as string[][])[0][0].slice(
          CLIPBOARD_PREFIX.length,
        ),
      );
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

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makeCopyEvent());
      });

      await vi.waitFor(() => {
        expect(mockWriteText).toHaveBeenCalledTimes(1);
      });

      const parsed = JSON.parse(
        (mockWriteText.mock.calls as string[][])[0][0].slice(
          CLIPBOARD_PREFIX.length,
        ),
      );
      expect(parsed.nodes).toHaveLength(1);
      expect(parsed.edges).toHaveLength(0);
    });

    it("copies nothing when no nodes are selected", async () => {
      const node = createTestNode("1", { selected: false });
      useNodeStore.setState({ nodes: [node] });

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makeCopyEvent());
      });

      await vi.waitFor(() => {
        expect(mockWriteText).toHaveBeenCalledTimes(1);
      });

      const parsed = JSON.parse(
        (mockWriteText.mock.calls as string[][])[0][0].slice(
          CLIPBOARD_PREFIX.length,
        ),
      );
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

      mockReadText.mockResolvedValue(clipboardPayload([node], []));

      useNodeStore.setState({ nodes: [], nodeCounter: 0 });

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makePasteEvent());
      });

      await vi.waitFor(() => {
        const { nodes } = useNodeStore.getState();
        expect(nodes).toHaveLength(1);
      });

      const { nodes } = useNodeStore.getState();
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

      mockReadText.mockResolvedValue(clipboardPayload([node], []));

      useNodeStore.setState({ nodes: [], nodeCounter: 0 });

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makePasteEvent());
      });

      await vi.waitFor(() => {
        const { nodes } = useNodeStore.getState();
        expect(nodes).toHaveLength(1);
      });

      const { nodes } = useNodeStore.getState();
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

      mockReadText.mockResolvedValue(clipboardPayload([nodeToPaste], []));

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makePasteEvent());
      });

      await vi.waitFor(() => {
        const { nodes } = useNodeStore.getState();
        expect(nodes).toHaveLength(2);
      });

      const { nodes } = useNodeStore.getState();
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

      mockReadText.mockResolvedValue(clipboardPayload([nodeA, nodeB], [edge]));

      useNodeStore.setState({ nodes: [], nodeCounter: 0 });
      useEdgeStore.setState({ edges: [] });

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makePasteEvent());
      });

      await vi.waitFor(() => {
        const { nodes } = useNodeStore.getState();
        expect(nodes).toHaveLength(2);
      });

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

    it("does nothing when clipboard does not have the expected prefix", async () => {
      mockReadText.mockResolvedValue("some random text");

      const existingNode = createTestNode("1", { position: { x: 0, y: 0 } });
      useNodeStore.setState({ nodes: [existingNode], nodeCounter: 0 });

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makePasteEvent());
      });

      // Give async operations time to settle
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

    it("does nothing when clipboard is empty", async () => {
      mockReadText.mockResolvedValue("");

      const existingNode = createTestNode("1", { position: { x: 0, y: 0 } });
      useNodeStore.setState({ nodes: [existingNode], nodeCounter: 0 });

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makePasteEvent());
      });

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

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makeCopyEvent());
      });

      // Clipboard write should NOT be called
      expect(mockWriteText).not.toHaveBeenCalled();

      document.body.removeChild(input);
    });

    it("ignores Ctrl+V when a textarea element is focused", async () => {
      mockReadText.mockResolvedValue(
        clipboardPayload(
          [createTestNode("a", { position: { x: 0, y: 0 } })],
          [],
        ),
      );

      useNodeStore.setState({ nodes: [], nodeCounter: 0 });

      const textarea = document.createElement("textarea");
      document.body.appendChild(textarea);
      textarea.focus();

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makePasteEvent());
      });

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

      const { result } = renderHook(() => useCopyPaste());

      act(() => {
        result.current(makeCopyEvent());
      });

      expect(mockWriteText).not.toHaveBeenCalled();

      document.body.removeChild(div);
    });
  });

  describe("meta key support (macOS)", () => {
    it("handles Cmd+C (metaKey) the same as Ctrl+C", async () => {
      const node = createTestNode("1", { selected: true });
      useNodeStore.setState({ nodes: [node] });

      const { result } = renderHook(() => useCopyPaste());

      const metaCopyEvent = new KeyboardEvent("keydown", {
        key: "c",
        metaKey: true,
        bubbles: true,
      });

      act(() => {
        result.current(metaCopyEvent);
      });

      await vi.waitFor(() => {
        expect(mockWriteText).toHaveBeenCalledTimes(1);
      });
    });

    it("handles Cmd+V (metaKey) the same as Ctrl+V", async () => {
      const node = createTestNode("orig", {
        selected: true,
        position: { x: 0, y: 0 },
      });
      mockReadText.mockResolvedValue(clipboardPayload([node], []));
      useNodeStore.setState({ nodes: [], nodeCounter: 0 });

      const { result } = renderHook(() => useCopyPaste());

      const metaPasteEvent = new KeyboardEvent("keydown", {
        key: "v",
        metaKey: true,
        bubbles: true,
      });

      act(() => {
        result.current(metaPasteEvent);
      });

      await vi.waitFor(() => {
        const { nodes } = useNodeStore.getState();
        expect(nodes).toHaveLength(1);
      });
    });
  });
});
