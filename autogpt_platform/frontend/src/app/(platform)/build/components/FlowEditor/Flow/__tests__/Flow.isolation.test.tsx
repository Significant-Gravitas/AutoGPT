import { act, render, screen } from "@testing-library/react";
import { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import type { CustomNode } from "../../nodes/CustomNode/CustomNode";
import { useNodeStore } from "../../../../stores/nodeStore";

// The real node store drives this test: a position or data write is what a
// drag frame or a keystroke in a node field does.

const renders = {
  reactFlow: 0,
  chat: 0,
  reviews: 0,
  safeMode: 0,
  draftRecovery: 0,
  controlPanel: 0,
};
let lastReactFlowNodes: unknown;

vi.mock("../useFlow", () => ({
  useFlow: () => ({
    onDragOver: vi.fn(),
    onDrop: vi.fn(),
    isFlowContentLoading: false,
    isInitialLoadComplete: true,
    isLocked: false,
    setIsLocked: vi.fn(),
    isReadOnly: false,
  }),
}));
vi.mock("../useFlowRealtime", () => ({ useFlowRealtime: vi.fn() }));
vi.mock("../useCopyPaste", () => ({ useCopyPaste: vi.fn() }));
vi.mock("../../edges/useCustomEdge", () => ({
  useCustomEdge: () => ({
    edges: [],
    onConnect: vi.fn(),
    onEdgesChange: vi.fn(),
  }),
}));
vi.mock("../../nodes/CustomNode/CustomNode", () => ({
  CustomNode: () => null,
}));
vi.mock("../../edges/CustomEdge", () => ({ default: () => null }));

vi.mock("@xyflow/react", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@xyflow/react")>()),
  ReactFlow: ({ children, nodes }: { children: ReactNode; nodes: unknown }) => {
    renders.reactFlow++;
    lastReactFlowNodes = nodes;
    return <div data-testid="react-flow">{children}</div>;
  },
  Background: () => null,
}));

vi.mock("@/app/api/__generated__/endpoints/graphs/graphs", () => ({
  useGetV1GetSpecificGraph: () => ({ data: { id: "graph-1" } }),
}));
vi.mock("nuqs", () => ({
  parseAsString: {},
  useQueryStates: () => [{ flowID: "graph-1", flowExecutionID: null }, vi.fn()],
}));

vi.mock("../../../NewControlPanel/NewControlPanel", () => ({
  default: () => {
    renders.controlPanel++;
    return null;
  },
}));
vi.mock("../../../BuilderActions/BuilderActions", () => ({
  BuilderActions: () => null,
}));
vi.mock("../components/CustomControl", () => ({
  CustomControls: () => null,
}));
vi.mock("../components/GraphLoadingBox", () => ({
  GraphLoadingBox: () => null,
}));
vi.mock("../../../FloatingSafeModeToogle", () => ({
  FloatingSafeModeToggle: () => {
    renders.safeMode++;
    return <div data-testid="safe-mode-toggle" />;
  },
}));
vi.mock("../../../DraftRecoveryDialog/DraftRecoveryPopup", () => ({
  DraftRecoveryPopup: () => {
    renders.draftRecovery++;
    return <div data-testid="draft-recovery" />;
  },
}));
vi.mock("../../../BuilderChatPanel/BuilderChatPanel", () => ({
  BuilderChatPanel: () => {
    renders.chat++;
    return <div data-testid="builder-chat" />;
  },
}));
vi.mock(
  "@/components/organisms/FloatingReviewsPanel/FloatingReviewsPanel",
  () => ({
    FloatingReviewsPanel: () => {
      renders.reviews++;
      return <div data-testid="reviews-panel" />;
    },
  }),
);

import { Flow } from "../Flow";

function makeNode(id: string, x: number): CustomNode {
  return {
    id,
    type: "custom",
    position: { x, y: 0 },
    data: {
      title: `Node ${id}`,
      block_id: "block",
      hardcodedValues: { text: "" },
      inputSchema: { type: "object", properties: {} },
      outputSchema: { type: "object", properties: {} },
      description: "",
      uiType: "Standard",
      costs: [],
      categories: [],
    },
  } as unknown as CustomNode;
}

beforeEach(() => {
  for (const key of Object.keys(renders) as (keyof typeof renders)[]) {
    renders[key] = 0;
  }
  useNodeStore.getState().setNodes([makeNode("1", 0), makeNode("2", 300)]);
});

afterEach(() => {
  useNodeStore.getState().setNodes([]);
});

describe("Flow canvas isolation", () => {
  test("a drag frame re-renders the canvas but not the panels beside it", () => {
    render(<Flow />);
    expect(screen.getByTestId("builder-chat")).toBeDefined();
    expect(screen.getByTestId("reviews-panel")).toBeDefined();
    expect(screen.getByTestId("safe-mode-toggle")).toBeDefined();
    expect(screen.getByTestId("draft-recovery")).toBeDefined();
    const mounted = { ...renders };

    for (let frame = 1; frame <= 5; frame++) {
      act(() => {
        useNodeStore.getState().onNodesChange([
          {
            type: "position",
            id: "1",
            position: { x: frame * 10, y: 0 },
            dragging: true,
          },
        ]);
      });
    }

    expect(renders.reactFlow).toBe(mounted.reactFlow + 5);
    expect(lastReactFlowNodes).toBe(useNodeStore.getState().nodes);
    expect(renders.chat).toBe(mounted.chat);
    expect(renders.reviews).toBe(mounted.reviews);
    expect(renders.safeMode).toBe(mounted.safeMode);
    expect(renders.draftRecovery).toBe(mounted.draftRecovery);
  });

  test("a field edit re-renders the canvas but not the panels beside it", () => {
    render(<Flow />);
    const mounted = { ...renders };

    act(() => {
      useNodeStore
        .getState()
        .updateNodeData("1", { hardcodedValues: { text: "hello" } });
    });

    expect(renders.reactFlow).toBe(mounted.reactFlow + 1);
    expect(renders.chat).toBe(mounted.chat);
    expect(renders.reviews).toBe(mounted.reviews);
  });

  test("controls inside the canvas still re-render with it", () => {
    render(<Flow />);
    const mounted = renders.controlPanel;

    act(() => {
      useNodeStore.getState().onNodesChange([
        {
          type: "position",
          id: "2",
          position: { x: 50, y: 50 },
          dragging: true,
        },
      ]);
    });

    expect(renders.controlPanel).toBe(mounted + 1);
  });
});
