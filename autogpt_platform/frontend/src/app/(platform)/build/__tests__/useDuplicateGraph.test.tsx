import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mockPush = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
}));

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

vi.mock("nuqs", () => ({
  parseAsString: {},
  useQueryStates: vi.fn(() => [{ flowID: "graph-1" }, vi.fn()]),
}));

let mockLibraryAgent: { id: string; graph_id: string } | undefined;
let mockIsLoading = false;
const mockForkAgent = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/library/library", () => ({
  useGetV2GetLibraryAgentByGraphId: vi.fn(() => ({
    data: mockLibraryAgent,
    isLoading: mockIsLoading,
  })),
  usePostV2ForkLibraryAgent: vi.fn(() => ({
    mutateAsync: mockForkAgent,
    isPending: false,
  })),
}));

import { useDuplicateGraph } from "../hooks/useDuplicateGraph";

describe("useDuplicateGraph", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLibraryAgent = undefined;
    mockIsLoading = false;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("cannot duplicate when there is no library agent for the graph", () => {
    mockLibraryAgent = undefined;

    const { result } = renderHook(() => useDuplicateGraph());

    expect(result.current.canDuplicate).toBe(false);
  });

  it("reports the library lookup as in flight while loading", () => {
    mockLibraryAgent = undefined;
    mockIsLoading = true;

    const { result } = renderHook(() => useDuplicateGraph());

    expect(result.current.isCheckingLibrary).toBe(true);
    expect(result.current.canDuplicate).toBe(false);
  });

  it("duplicates via fork and navigates to the new graph", async () => {
    mockLibraryAgent = { id: "lib-1", graph_id: "graph-1" };
    mockForkAgent.mockResolvedValue({
      status: 200,
      data: { id: "lib-2", graph_id: "graph-2" },
    });

    const { result } = renderHook(() => useDuplicateGraph());

    expect(result.current.canDuplicate).toBe(true);

    await act(async () => {
      await result.current.duplicate();
    });

    expect(mockForkAgent).toHaveBeenCalledWith({ libraryAgentId: "lib-1" });
    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith("/build?flowID=graph-2");
    });
  });

  it("does nothing when there is no library agent to fork", async () => {
    mockLibraryAgent = undefined;

    const { result } = renderHook(() => useDuplicateGraph());

    await act(async () => {
      await result.current.duplicate();
    });

    expect(mockForkAgent).not.toHaveBeenCalled();
    expect(mockPush).not.toHaveBeenCalled();
  });

  it("shows a toast and does not navigate when the fork returns no graph", async () => {
    mockLibraryAgent = { id: "lib-1", graph_id: "graph-1" };
    mockForkAgent.mockResolvedValue({ status: 200, data: { id: "lib-2" } });

    const { result } = renderHook(() => useDuplicateGraph());

    await act(async () => {
      await result.current.duplicate();
    });

    expect(mockPush).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  it("shows a toast and does not navigate when duplicate fails", async () => {
    mockLibraryAgent = { id: "lib-1", graph_id: "graph-1" };
    mockForkAgent.mockRejectedValue(new Error("boom"));

    const { result } = renderHook(() => useDuplicateGraph());

    await act(async () => {
      await result.current.duplicate();
    });

    expect(mockPush).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  it("falls back to a generic message for non-Error exceptions", async () => {
    mockLibraryAgent = { id: "lib-1", graph_id: "graph-1" };
    mockForkAgent.mockRejectedValue("string error");

    const { result } = renderHook(() => useDuplicateGraph());

    await act(async () => {
      await result.current.duplicate();
    });

    expect(mockPush).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        variant: "destructive",
        description: "An unexpected error occurred.",
      }),
    );
  });
});
