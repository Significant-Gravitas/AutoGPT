import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, renderHook, screen } from "@testing-library/react";
import type { ToastProps } from "@/components/molecules/Toast/use-toast";

const mockDismiss = vi.fn();
const mockToast = vi.fn((_props: ToastProps) => ({ dismiss: mockDismiss }));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (props: ToastProps) => mockToast(props),
}));

const mockGetViewport = vi.fn(() => ({ x: 0, y: 0, zoom: 1 }));
const flowState = { width: 1000, height: 800 };
const mockFitView = vi.fn();

vi.mock("@xyflow/react", () => ({
  useReactFlow: () => ({ getViewport: mockGetViewport, fitView: mockFitView }),
  useStoreApi: () => ({ getState: () => flowState }),
  useStore: (selector: (state: { width: number; height: number }) => number) =>
    selector(flowState),
}));

const mockAddBlock = vi.fn((block, _hv, position) => {
  const node = {
    id: `node-${nodeState.nodes.length + 1}`,
    position,
    data: { uiType: block.uiType },
  };
  nodeState.nodes = [...nodeState.nodes, node];
  return node;
});

const nodeState = {
  addBlock: mockAddBlock,
  nodes: [] as Array<{
    position: { x: number; y: number };
    width?: number;
    measured?: { width: number; height: number };
    data: { uiType: string };
  }>,
};

vi.mock("@/app/(platform)/build/stores/nodeStore", () => ({
  useNodeStore: Object.assign(
    (selector: (state: typeof nodeState) => unknown) => selector(nodeState),
    { getState: () => nodeState },
  ),
}));

import { useAddBlockToBuilder } from "../useAddBlockToBuilder";
import { BlockUIType } from "@/app/(platform)/build/components/types";

function makeBlock(uiType = BlockUIType.STANDARD) {
  return {
    id: "test-block",
    name: "Test Block",
    description: "",
    inputSchema: {},
    outputSchema: {},
    uiType,
  } as Parameters<
    ReturnType<typeof useAddBlockToBuilder>["addBlockWithPlacement"]
  >[0];
}

describe("useAddBlockToBuilder", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    nodeState.nodes = [];
    flowState.width = 1000;
    flowState.height = 800;
    mockGetViewport.mockReturnValue({ x: 0, y: 0, zoom: 1 });
  });

  it("places a block on an empty canvas", () => {
    const { result } = renderHook(() => useAddBlockToBuilder());

    result.current.addBlockWithPlacement(makeBlock());

    expect(mockAddBlock).toHaveBeenCalledOnce();
    const position = mockAddBlock.mock.calls[0][2];
    expect(position).toEqual({ x: 70, y: 70 });
  });

  it("avoids overlapping existing nodes", () => {
    nodeState.nodes = [
      {
        position: { x: 70, y: 70 },
        measured: { width: 400, height: 400 },
        data: { uiType: BlockUIType.STANDARD },
      },
    ];

    const { result } = renderHook(() => useAddBlockToBuilder());
    result.current.addBlockWithPlacement(makeBlock());

    const position = mockAddBlock.mock.calls[0][2];
    expect(position.x).not.toBe(70);
  });

  it("uses the newly inserted node when two blocks are added before rerender", () => {
    const { result } = renderHook(() => useAddBlockToBuilder());

    const first = result.current.addBlockWithPlacement(makeBlock());
    const second = result.current.addBlockWithPlacement(makeBlock());

    expect(second.position).not.toEqual(first.position);
    expect(mockFitView).not.toHaveBeenCalled();
  });

  it("reads current canvas dimensions when adding after a resize", () => {
    const { result } = renderHook(() => useAddBlockToBuilder());
    flowState.width = 440;
    flowState.height = 490;

    const node = result.current.addBlockWithPlacement(makeBlock());

    expect(node.position.x).toBeGreaterThanOrEqual(40);
    expect(node.position.x + 350).toBeLessThanOrEqual(400);
    expect(node.position.y + 400).toBeLessThanOrEqual(450);
  });

  it("uses smaller dimensions for note blocks", () => {
    const { result } = renderHook(() => useAddBlockToBuilder());

    result.current.addBlockWithPlacement(makeBlock(BlockUIType.NOTE));

    expect(mockAddBlock).toHaveBeenCalledOnce();
  });

  it("passes hardcoded values through to addBlock", () => {
    const { result } = renderHook(() => useAddBlockToBuilder());
    const hardcoded = { key: "value" };

    result.current.addBlockWithPlacement(makeBlock(), hardcoded);

    expect(mockAddBlock).toHaveBeenCalledWith(
      expect.anything(),
      hardcoded,
      expect.any(Object),
    );
  });

  it("moves the camera only when the user chooses Show block", () => {
    flowState.width = 300;
    const { result } = renderHook(() => useAddBlockToBuilder());
    const node = result.current.addBlockWithPlacement(makeBlock());

    expect(mockFitView).not.toHaveBeenCalled();
    render(<>{mockToast.mock.calls[0][0].action}</>);
    fireEvent.click(screen.getByRole("button", { name: "Show block" }));

    expect(mockFitView).toHaveBeenCalledWith(
      expect.objectContaining({ nodes: [{ id: node.id }], maxZoom: 1 }),
    );
    expect(mockDismiss).toHaveBeenCalledOnce();
  });

  it("ignores Show block after the inserted block has been removed", () => {
    flowState.width = 300;
    const { result } = renderHook(() => useAddBlockToBuilder());
    result.current.addBlockWithPlacement(makeBlock());
    nodeState.nodes = [];
    render(<>{mockToast.mock.calls[0][0].action}</>);

    fireEvent.click(screen.getByRole("button", { name: "Show block" }));

    expect(mockFitView).not.toHaveBeenCalled();
  });

  it("uses the latest pan and zoom at insertion", () => {
    const { result } = renderHook(() => useAddBlockToBuilder());
    mockGetViewport.mockReturnValue({ x: -1000, y: -500, zoom: 2 });

    const node = result.current.addBlockWithPlacement(makeBlock());

    expect(node.position.x).toBeGreaterThanOrEqual(520);
    expect(node.position.y).toBeGreaterThanOrEqual(270);
  });
});
