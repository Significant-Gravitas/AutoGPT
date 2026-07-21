import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

const mockSetQueryStates = vi.fn();
vi.mock("nuqs", () => ({
  parseAsString: {},
  parseAsInteger: {},
  useQueryStates: vi.fn(() => [
    { flowID: "graph-1", flowVersion: 1 },
    mockSetQueryStates,
  ]),
}));

// The graph currently loaded for the flow — swapped per test to exercise the
// update, create, and no-change branches.
let mockCurrentGraph: unknown;
const mockCreate = vi.fn();
const mockUpdate = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/graphs/graphs", () => ({
  useGetV1GetSpecificGraph: () => ({ data: mockCurrentGraph }),
  usePostV1CreateNewGraph: () => ({
    mutateAsync: mockCreate,
    isPending: false,
  }),
  usePutV1UpdateGraphVersion: () => ({
    mutateAsync: mockUpdate,
    isPending: false,
  }),
}));

vi.mock("@/app/(platform)/build/stores/nodeStore", () => ({
  useNodeStore: { getState: () => ({ getBackendNodes: () => [] }) },
}));
vi.mock("@/app/(platform)/build/stores/edgeStore", () => ({
  useEdgeStore: { getState: () => ({ getBackendLinks: () => [] }) },
}));

const mockSetGraphSchemas = vi.fn();
vi.mock("@/app/(platform)/build/stores/graphStore", () => ({
  useGraphStore: (selector: (s: { setGraphSchemas: unknown }) => unknown) =>
    selector({ setGraphSchemas: mockSetGraphSchemas }),
}));

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

import { useSaveGraph } from "../hooks/useSaveGraph";

describe("useSaveGraph", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCurrentGraph = undefined;
    mockEquivalent = false;
  });

  afterEach(() => {
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
    mockUpdate.mockResolvedValue({ data: updated });

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

  it("returns the created graph model when there is no existing graph", async () => {
    mockCurrentGraph = undefined;
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
});
