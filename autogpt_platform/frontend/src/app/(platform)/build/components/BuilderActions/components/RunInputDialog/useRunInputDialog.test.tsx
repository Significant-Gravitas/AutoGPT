import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Stale query state: the URL still points at the pre-save version (flowVersion:
// 1). A save right before opening the dialog bumps the real version to 2, but
// setQueryStates updates the URL asynchronously — so it can still read 1 here.
const mockSetQueryStates = vi.fn();
vi.mock("nuqs", () => ({
  parseAsString: {},
  parseAsInteger: {},
  useQueryStates: vi.fn(() => [
    { flowID: "graph-1", flowVersion: 1, flowExecutionID: null },
    mockSetQueryStates,
  ]),
}));

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

const mockExecuteGraph = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/graphs/graphs", () => ({
  usePostV1ExecuteGraphAgent: () => ({
    mutateAsync: mockExecuteGraph,
    isPending: false,
  }),
}));

vi.mock("@/lib/autogpt-server-api", () => {
  class MockApiError extends Error {
    isGraphValidationError() {
      return false;
    }
  }
  return {
    ApiError: MockApiError,
    CredentialsMetaInput: {},
    GraphExecutionMeta: {},
  };
});

const graphState = {
  credentialsInputSchema: undefined,
  setIsGraphRunning: vi.fn(),
};
vi.mock("@/app/(platform)/build/stores/graphStore", () => ({
  useGraphStore: (selector: (s: typeof graphState) => unknown) =>
    selector(graphState),
}));

const nodeState = {
  clearAllNodeExecutionResults: vi.fn(),
  cleanNodesStatuses: vi.fn(),
  updateNodeErrors: vi.fn(),
  nodes: [],
};
vi.mock("@/app/(platform)/build/stores/nodeStore", () => ({
  useNodeStore: Object.assign(
    (selector: (s: typeof nodeState) => unknown) => selector(nodeState),
    { getState: () => nodeState },
  ),
}));

vi.mock("@xyflow/react", () => ({
  useReactFlow: () => ({ setViewport: vi.fn() }),
}));

import { useRunInputDialog } from "./useRunInputDialog";

describe("useRunInputDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("runs the version passed in, not the stale one in the URL", async () => {
    const { result } = renderHook(() =>
      useRunInputDialog({
        setIsOpen: vi.fn(),
        graphID: "graph-1",
        graphVersion: 2,
      }),
    );

    await act(async () => {
      await result.current.handleManualRun();
    });

    expect(mockExecuteGraph).toHaveBeenCalledWith(
      expect.objectContaining({ graphId: "graph-1", graphVersion: 2 }),
    );
  });

  it("falls back to the URL version when no version is passed in", async () => {
    const { result } = renderHook(() =>
      useRunInputDialog({ setIsOpen: vi.fn() }),
    );

    await act(async () => {
      await result.current.handleManualRun();
    });

    expect(mockExecuteGraph).toHaveBeenCalledWith(
      expect.objectContaining({ graphId: "graph-1", graphVersion: 1 }),
    );
  });
});
