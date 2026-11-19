import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

// The workflow in the URL; null until its first save.
let mockFlowID: string | null;
let mockFlowVersion: number | null;
const mockSetQueryStates = vi.fn();
vi.mock("nuqs", () => ({
  parseAsString: {},
  parseAsInteger: {},
  useQueryStates: vi.fn(() => [
    { flowID: mockFlowID, flowVersion: mockFlowVersion },
    mockSetQueryStates,
  ]),
}));

// The saved graph the canvas shows — swapped per test to exercise the update
// and no-change branches.
let mockCurrentGraph: unknown;
const mockCreate = vi.fn();
const mockUpdate = vi.fn();
let mockUpdateOptions: {
  mutation: {
    onSuccess: (response: unknown, variables: unknown) => Promise<void>;
  };
};
vi.mock("@/app/api/__generated__/endpoints/graphs/graphs", () => ({
  usePostV1CreateNewGraph: () => ({
    mutateAsync: mockCreate,
    isPending: false,
  }),
  usePutV1UpdateGraphVersion: (options: typeof mockUpdateOptions) => {
    mockUpdateOptions = options;
    return { mutateAsync: mockUpdate, isPending: false };
  },
}));

const mockSetNodeErrorsForBackendId = vi.fn();
const mockClearAllNodeErrors = vi.fn();
vi.mock("@/app/(platform)/build/stores/nodeStore", () => ({
  useNodeStore: {
    getState: () => ({
      getBackendNodes: () => [],
      setNodeErrorsForBackendId: mockSetNodeErrorsForBackendId,
      clearAllNodeErrors: mockClearAllNodeErrors,
    }),
  },
}));
vi.mock("@/app/(platform)/build/stores/edgeStore", () => ({
  useEdgeStore: { getState: () => ({ getBackendLinks: () => [] }) },
}));

const mockSetGraphSchemas = vi.fn();
// Kept so a test can load the canvas while a save waits for it.
const mockGraphStoreListeners = new Set<(state: unknown) => void>();
let mockLoadController: AbortController;
vi.mock("@/app/(platform)/build/stores/graphStore", () => ({
  useGraphStore: Object.assign(
    (selector: (s: { setGraphSchemas: unknown }) => unknown) =>
      selector({ setGraphSchemas: mockSetGraphSchemas }),
    {
      getState: () => mockGraphStoreState(),
      subscribe: (listener: (state: unknown) => void) => {
        mockGraphStoreListeners.add(listener);
        return () => mockGraphStoreListeners.delete(listener);
      },
    },
  ),
}));

function mockGraphStoreState() {
  return {
    loadedGraph: mockCurrentGraph ?? null,
    loadedForVersion: 1,
    loadSignal: mockLoadController.signal,
    loadFailed: false,
  };
}

function loadCanvas(graph: unknown) {
  mockCurrentGraph = graph;
  mockGraphStoreListeners.forEach((listener) =>
    listener(mockGraphStoreState()),
  );
}

let mockEquivalent = false;
vi.mock(
  "@/app/(platform)/build/components/NewControlPanel/NewSaveControl/helpers",
  () => ({
    graphsEquivalent: () => mockEquivalent,
  }),
);

vi.mock("@/services/builder-draft/draft-service", () => ({
  draftService: { deleteDraft: vi.fn().mockResolvedValue(undefined) },
  getTempFlowId: () => null,
  clearTempFlowId: vi.fn(),
}));

import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { draftService } from "@/services/builder-draft/draft-service";
import { useSaveGraph } from "../hooks/useSaveGraph";

describe("useSaveGraph", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGraphStoreListeners.clear();
    mockLoadController = new AbortController();
    mockFlowID = "graph-1";
    mockFlowVersion = 1;
    mockCurrentGraph = undefined;
    mockEquivalent = false;
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("returns the updated graph model when saving changes to an existing graph", async () => {
    mockCurrentGraph = { id: "graph-1", name: "Agent", version: 1 };
    const updated = {
      id: "graph-1",
      version: 2,
      input_schema: { properties: {} },
      credentials_input_schema: { properties: {} },
      output_schema: { properties: {} },
    };
    mockUpdate.mockResolvedValue({ data: { graph: updated } });

    const { result } = renderHook(() => useSaveGraph({ showToast: false }));

    let returned: unknown;
    await act(async () => {
      returned = await result.current.saveGraph();
    });

    expect(mockUpdate).toHaveBeenCalledWith({
      graphId: "graph-1",
      data: expect.objectContaining({ id: "graph-1" }),
    });
    expect(mockSetGraphSchemas).toHaveBeenCalled();
    expect(returned).toBe(updated);
  });

  it("clears the draft under the sent ID when the save answers with a copy", async () => {
    mockCurrentGraph = { id: "graph-1", name: "Agent", version: 1 };
    const copy = { id: "graph-2", version: 1 };
    mockUpdate.mockImplementation(async (variables) => {
      const response = { data: { graph: copy, skipped_webhook_presets: [] } };
      await mockUpdateOptions.mutation.onSuccess(response, variables);
      return response;
    });

    const { result } = renderHook(() => useSaveGraph({ showToast: false }));
    await act(async () => {
      await result.current.saveGraph();
    });

    expect(
      vi.mocked(draftService.deleteDraft).mock.calls.flat().sort(),
    ).toEqual(["graph-1", "graph-2"]);
  });

  it("warns the user when a trigger swap leaves webhook presets behind", async () => {
    mockCurrentGraph = { id: "graph-1", name: "Agent", version: 1 };
    const updated = {
      id: "graph-1",
      version: 2,
      input_schema: { properties: {} },
      credentials_input_schema: { properties: {} },
      output_schema: { properties: {} },
    };
    mockUpdate.mockResolvedValue({
      data: {
        graph: updated,
        skipped_webhook_presets: [
          { id: "preset-1", name: "GitHub PR trigger", pinned_version: 1 },
        ],
      },
    });

    const { result } = renderHook(() => useSaveGraph({ showToast: false }));

    await act(async () => {
      await result.current.saveGraph();
    });

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        variant: "info",
        description: expect.stringContaining("GitHub PR trigger"),
      }),
    );
  });

  it("does not warn when no webhook presets are skipped", async () => {
    mockCurrentGraph = { id: "graph-1", name: "Agent", version: 1 };
    const updated = {
      id: "graph-1",
      version: 2,
      input_schema: { properties: {} },
      credentials_input_schema: { properties: {} },
      output_schema: { properties: {} },
    };
    mockUpdate.mockResolvedValue({
      data: { graph: updated, skipped_webhook_presets: [] },
    });

    const { result } = renderHook(() => useSaveGraph({ showToast: false }));

    await act(async () => {
      await result.current.saveGraph();
    });

    expect(mockToast).not.toHaveBeenCalled();
  });

  it("returns the created graph model when the workflow has not been saved yet", async () => {
    mockFlowID = null;
    mockFlowVersion = null;
    const created = {
      id: "new-graph",
      version: 1,
      input_schema: { properties: {} },
      credentials_input_schema: { properties: {} },
      output_schema: { properties: {} },
    };
    mockCreate.mockResolvedValue({ data: created });

    const { result } = renderHook(() => useSaveGraph({ showToast: false }));

    let returned: unknown;
    await act(async () => {
      returned = await result.current.saveGraph();
    });

    expect(mockCreate).toHaveBeenCalled();
    expect(mockUpdate).not.toHaveBeenCalled();
    expect(returned).toBe(created);
  });

  it("returns the current graph without saving when nothing changed", async () => {
    mockCurrentGraph = { id: "graph-1", name: "Agent", version: 1 };
    mockEquivalent = true;

    const { result } = renderHook(() => useSaveGraph({ showToast: false }));

    let returned: unknown;
    await act(async () => {
      returned = await result.current.saveGraph();
    });

    expect(mockUpdate).not.toHaveBeenCalled();
    expect(mockCreate).not.toHaveBeenCalled();
    expect(returned).toBe(mockCurrentGraph);
  });

  it("names the invalid field and marks it on its node when the backend refuses the graph", async () => {
    mockFlowID = null;
    mockFlowVersion = null;
    mockCreate.mockRejectedValue(
      new ApiError("Agent Output › Name: This field is required", 400, {
        detail: {
          type: "validation_error",
          message: "Agent Output › Name: This field is required",
          node_errors: { "node-2": { name: "This field is required" } },
        },
      }),
    );

    const { result } = renderHook(() => useSaveGraph({ showToast: true }));

    let returned: unknown;
    await act(async () => {
      returned = await result.current.saveGraph();
    });

    expect(returned).toBeNull();
    expect(mockClearAllNodeErrors).toHaveBeenCalledBefore(
      mockSetNodeErrorsForBackendId,
    );
    expect(mockSetNodeErrorsForBackendId).toHaveBeenCalledWith("node-2", {
      name: "This field is required",
    });
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Couldn't save your workflow",
        description: expect.stringContaining(
          "Agent Output › Name: This field is required",
        ),
        variant: "destructive",
      }),
    );
  });

  it("tells the user to try again when the canvas never finishes loading the saved workflow", async () => {
    vi.useFakeTimers();

    const { result } = renderHook(() => useSaveGraph({ showToast: false }));

    let returned: unknown;
    await act(async () => {
      const saving = result.current.saveGraph();
      await vi.advanceTimersByTimeAsync(30_000);
      returned = await saving;
    });

    expect(returned).toBeNull();
    expect(mockCreate).not.toHaveBeenCalled();
    expect(mockUpdate).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Error saving workflow",
        description: "Your workflow is still loading. Try again in a moment.",
      }),
    );
  });

  it("waits until the canvas shows this version of the saved workflow, then saves it", async () => {
    const updated = {
      id: "graph-1",
      version: 2,
      input_schema: { properties: {} },
      credentials_input_schema: { properties: {} },
      output_schema: { properties: {} },
    };
    mockUpdate.mockResolvedValue({ data: { graph: updated } });
    const { result } = renderHook(() => useSaveGraph({ showToast: false }));

    let saving: Promise<unknown> = Promise.resolve();
    act(() => {
      saving = result.current.saveGraph();
    });
    expect(result.current.isSaving).toBe(true);

    await act(async () => {
      loadCanvas({ id: "graph-1", name: "Agent", version: 0 });
    });
    expect(mockUpdate).not.toHaveBeenCalled();

    let returned: unknown;
    await act(async () => {
      loadCanvas({ id: "graph-1", name: "Agent", version: 1 });
      returned = await saving;
    });
    expect(mockUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ graphId: "graph-1" }),
    );
    expect(returned).toBe(updated);
    expect(result.current.isSaving).toBe(false);
  });

  it("stays saving until every save waiting for the canvas has finished", async () => {
    mockEquivalent = true;
    const { result, rerender } = renderHook(() =>
      useSaveGraph({ showToast: false }),
    );

    let first: Promise<unknown> = Promise.resolve();
    act(() => {
      first = result.current.saveGraph();
    });
    mockFlowVersion = 2;
    rerender();
    let second: Promise<unknown> = Promise.resolve();
    act(() => {
      second = result.current.saveGraph();
    });

    await act(async () => {
      loadCanvas({ id: "graph-1", name: "Agent", version: 1 });
      await first;
    });
    expect(result.current.isSaving).toBe(true);

    await act(async () => {
      loadCanvas({ id: "graph-1", name: "Agent", version: 2 });
      await second;
    });
    expect(result.current.isSaving).toBe(false);
  });

  it("gives up without telling the user when the builder leaves the workflow before it loads", async () => {
    const { result } = renderHook(() => useSaveGraph({ showToast: true }));

    let returned: unknown;
    await act(async () => {
      const saving = result.current.saveGraph();
      mockLoadController.abort();
      returned = await saving;
    });

    expect(returned).toBeNull();
    expect(mockToast).not.toHaveBeenCalled();
    expect(mockCreate).not.toHaveBeenCalled();
    expect(mockUpdate).not.toHaveBeenCalled();
  });

  it("reports any other failed save instead of throwing it", async () => {
    mockCurrentGraph = { id: "graph-1", name: "Agent", version: 1 };
    mockUpdate.mockRejectedValue(new Error("Network unreachable"));

    const { result } = renderHook(() => useSaveGraph({ showToast: false }));

    let returned: unknown;
    await act(async () => {
      returned = await result.current.saveGraph();
    });

    expect(returned).toBeNull();
    expect(mockSetNodeErrorsForBackendId).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Error saving workflow",
        description: "Network unreachable",
      }),
    );
  });
});
