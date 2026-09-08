import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Stale query state: the URL still points at the version that existed BEFORE
// this run's save (flowVersion: 1). The save below bumps it to version 2.
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

// saveGraph returns the freshly-saved graph model (new version) — mirroring the
// real hook. The regression is that the run must use THIS version, not the
// stale closure value.
const mockSaveGraph = vi.fn();
vi.mock("@/app/(platform)/build/hooks/useSaveGraph", () => ({
  useSaveGraph: () => ({ saveGraph: mockSaveGraph, isSaving: false }),
}));

const mockExecuteGraph = vi.fn();
const mockStopGraph = vi.fn();
const mockSendDatafastEvent = vi.hoisted(() => vi.fn());
vi.mock("@/services/analytics", () => ({
  analytics: { sendDatafastEvent: mockSendDatafastEvent },
}));

interface ExecutionVariables {
  data: { dry_run?: boolean };
}

interface ExecutionOptions {
  mutation: {
    onSuccess: (
      response: { status: number; data: Record<string, unknown> },
      variables: ExecutionVariables,
    ) => void;
  };
}

vi.mock("@/app/api/__generated__/endpoints/graphs/graphs", () => ({
  usePostV1ExecuteGraphAgent: (options: ExecutionOptions) => ({
    async mutateAsync(variables: ExecutionVariables) {
      const response = await mockExecuteGraph(variables);
      options.mutation.onSuccess(response, variables);
      return response;
    },
    isPending: false,
  }),
  usePostV1StopGraphExecution: () => ({
    mutateAsync: mockStopGraph,
    isPending: false,
  }),
}));

const graphState = {
  hasInputs: () => false,
  hasCredentials: () => false,
  setIsGraphRunning: vi.fn(),
};
vi.mock("@/app/(platform)/build/stores/graphStore", () => ({
  useGraphStore: (selector: (s: typeof graphState) => unknown) =>
    selector(graphState),
}));

const nodeState = {
  setNodeErrorsForBackendId: vi.fn(),
  clearAllNodeErrors: vi.fn(),
  cleanNodesStatuses: vi.fn(),
  clearAllNodeExecutionResults: vi.fn(),
  nodes: [],
  updateNodeErrors: vi.fn(),
};
vi.mock("@/app/(platform)/build/stores/nodeStore", () => ({
  useNodeStore: Object.assign(
    (selector: (s: typeof nodeState) => unknown) => selector(nodeState),
    { getState: () => nodeState },
  ),
}));

const tutorialState = {
  forceOpenRunInputDialog: false,
  setForceOpenRunInputDialog: vi.fn(),
};
vi.mock("@/app/(platform)/build/stores/tutorialStore", () => ({
  useTutorialStore: (selector: (s: typeof tutorialState) => unknown) =>
    selector(tutorialState),
}));

import { useRunGraph } from "./useRunGraph";

describe("useRunGraph", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockExecuteGraph.mockResolvedValue({
      status: 200,
      data: { id: "run-1", graph_id: "graph-1" },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("runs the version returned by the save, not the stale closure version", async () => {
    // Save creates a new version (2) even though the closure still holds v1.
    mockSaveGraph.mockResolvedValue({ id: "graph-1", version: 2 });

    const { result } = renderHook(() => useRunGraph());

    await act(async () => {
      await result.current.handleRunGraph();
    });

    expect(mockExecuteGraph).toHaveBeenCalledWith(
      expect.objectContaining({ graphId: "graph-1", graphVersion: 2 }),
    );
  });

  it("runs the unchanged version when the save reports no changes", async () => {
    // No new version created — saveGraph resolves with the existing graph
    // (mirrors useSaveGraph's no-op branch, which returns `graph`, not
    // undefined). The run must target that same, matching version.
    mockSaveGraph.mockResolvedValue({ id: "graph-1", version: 1 });

    const { result } = renderHook(() => useRunGraph());

    await act(async () => {
      await result.current.handleRunGraph();
    });

    expect(mockExecuteGraph).toHaveBeenCalledWith(
      expect.objectContaining({ graphId: "graph-1", graphVersion: 1 }),
    );
  });

  it("records a successful real run with its builder surface", async () => {
    mockSaveGraph.mockResolvedValue({ id: "graph-1", version: 1 });
    const { result } = renderHook(() => useRunGraph());

    await act(async () => {
      await result.current.handleRunGraph();
    });

    expect(mockSendDatafastEvent).toHaveBeenCalledExactlyOnceWith("run_agent", {
      id: "graph-1",
      name: "",
      surface: "builder",
    });
  });

  it("does not count a simulation as an activation", async () => {
    mockSaveGraph.mockResolvedValue({ id: "graph-1", version: 1 });
    const { result } = renderHook(() => useRunGraph());

    await act(async () => {
      await result.current.handleRunGraph({ dryRun: true });
    });

    expect(mockSendDatafastEvent).not.toHaveBeenCalled();
  });

  it("does not count an API error response as an activation", async () => {
    mockSaveGraph.mockResolvedValue({ id: "graph-1", version: 1 });
    mockExecuteGraph.mockResolvedValue({
      status: 422,
      data: { detail: "Invalid graph" },
    });
    const { result } = renderHook(() => useRunGraph());

    await act(async () => {
      await result.current.handleRunGraph();
    });

    expect(mockSendDatafastEvent).not.toHaveBeenCalled();
  });
});
