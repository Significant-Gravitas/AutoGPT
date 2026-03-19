import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

const mockScreenToFlowPosition = vi.fn((pos: { x: number; y: number }) => pos);
const mockFitView = vi.fn();

vi.mock("@xyflow/react", async () => {
  const actual = await vi.importActual("@xyflow/react");
  return {
    ...actual,
    useReactFlow: () => ({
      screenToFlowPosition: mockScreenToFlowPosition,
      fitView: mockFitView,
    }),
  };
});

const mockSetQueryStates = vi.fn();
let mockQueryStateValues: {
  flowID: string | null;
  flowVersion: number | null;
  flowExecutionID: string | null;
} = {
  flowID: null,
  flowVersion: null,
  flowExecutionID: null,
};

vi.mock("nuqs", () => ({
  parseAsString: {},
  parseAsInteger: {},
  useQueryStates: vi.fn(() => [mockQueryStateValues, mockSetQueryStates]),
}));

let mockGraphLoading = false;
let mockBlocksLoading = false;

vi.mock("@/app/api/__generated__/endpoints/graphs/graphs", () => ({
  useGetV1GetSpecificGraph: vi.fn(() => ({
    data: undefined,
    isLoading: mockGraphLoading,
  })),
  useGetV1GetExecutionDetails: vi.fn(() => ({
    data: undefined,
  })),
  useGetV1ListUserGraphs: vi.fn(() => ({
    data: undefined,
  })),
}));

vi.mock("@/app/api/__generated__/endpoints/default/default", () => ({
  useGetV2GetSpecificBlocks: vi.fn(() => ({
    data: undefined,
    isLoading: mockBlocksLoading,
  })),
}));

vi.mock("@/app/api/helpers", () => ({
  okData: (res: { data: unknown }) => res?.data,
}));

vi.mock("../components/helper", () => ({
  convertNodesPlusBlockInfoIntoCustomNodes: vi.fn(),
}));

describe("useFlow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers({ shouldAdvanceTime: true });
    mockGraphLoading = false;
    mockBlocksLoading = false;
    mockQueryStateValues = {
      flowID: null,
      flowVersion: null,
      flowExecutionID: null,
    };
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe("loading states", () => {
    it("returns isFlowContentLoading true when graph is loading", async () => {
      mockGraphLoading = true;
      mockQueryStateValues = {
        flowID: "test-flow",
        flowVersion: 1,
        flowExecutionID: null,
      };

      const { useFlow } = await import("../components/FlowEditor/Flow/useFlow");
      const { result } = renderHook(() => useFlow());

      expect(result.current.isFlowContentLoading).toBe(true);
    });

    it("returns isFlowContentLoading true when blocks are loading", async () => {
      mockBlocksLoading = true;
      mockQueryStateValues = {
        flowID: "test-flow",
        flowVersion: 1,
        flowExecutionID: null,
      };

      const { useFlow } = await import("../components/FlowEditor/Flow/useFlow");
      const { result } = renderHook(() => useFlow());

      expect(result.current.isFlowContentLoading).toBe(true);
    });

    it("returns isFlowContentLoading false when neither is loading", async () => {
      const { useFlow } = await import("../components/FlowEditor/Flow/useFlow");
      const { result } = renderHook(() => useFlow());

      expect(result.current.isFlowContentLoading).toBe(false);
    });
  });

  describe("initial load completion", () => {
    it("marks initial load complete for new flows without flowID", async () => {
      const { useFlow } = await import("../components/FlowEditor/Flow/useFlow");
      const { result } = renderHook(() => useFlow());

      expect(result.current.isInitialLoadComplete).toBe(false);

      await act(async () => {
        vi.advanceTimersByTime(300);
      });

      expect(result.current.isInitialLoadComplete).toBe(true);
    });
  });
});
