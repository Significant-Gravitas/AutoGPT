import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";

const mockGetViewport = vi.fn(() => ({ x: 0, y: 0, zoom: 1 }));

vi.mock("@xyflow/react", () => ({
  useReactFlow: () => ({ getViewport: mockGetViewport }),
  useStore: (selector: (state: { width: number; height: number }) => number) =>
    selector({ width: 1000, height: 800 }),
}));

const mockAddBlock = vi.fn((_block, _hv, position) => ({
  id: "node-1",
  position,
  data: { metadata: {} },
}));

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
  useNodeStore: (selector: (state: typeof nodeState) => unknown) =>
    selector(nodeState),
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
});
