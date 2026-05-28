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

let mockUser: { id: string } | null = null;
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ user: mockUser }),
}));

vi.mock("nuqs", () => ({
  parseAsString: {},
  useQueryStates: vi.fn(() => [{ flowID: "graph-1" }, vi.fn()]),
}));

let mockGraph: { id: string; user_id: string } | undefined;
vi.mock("@/app/api/__generated__/endpoints/graphs/graphs", () => ({
  useGetV1GetSpecificGraph: vi.fn(() => ({ data: mockGraph })),
}));

let mockLibraryAgent: { id: string; graph_id: string } | undefined;
const mockForkAgent = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/library/library", () => ({
  useGetV2GetLibraryAgentByGraphId: vi.fn(() => ({ data: mockLibraryAgent })),
  usePostV2ForkLibraryAgent: vi.fn(() => ({
    mutateAsync: mockForkAgent,
    isPending: false,
  })),
}));

import { useIsReadOnlyGraph } from "../hooks/useIsReadOnlyGraph";

describe("useIsReadOnlyGraph", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUser = { id: "user-1" };
    mockGraph = { id: "graph-1", user_id: "user-1" };
    mockLibraryAgent = undefined;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("is not read-only when the current user owns the graph", () => {
    mockGraph = { id: "graph-1", user_id: "user-1" };

    const { result } = renderHook(() => useIsReadOnlyGraph());

    expect(result.current.isReadOnly).toBe(false);
  });

  it("is read-only when the graph is owned by a different user", () => {
    mockGraph = { id: "graph-1", user_id: "other-user" };

    const { result } = renderHook(() => useIsReadOnlyGraph());

    expect(result.current.isReadOnly).toBe(true);
  });

  it("is not read-only while the graph or user is still loading", () => {
    mockGraph = undefined;

    const { result } = renderHook(() => useIsReadOnlyGraph());

    expect(result.current.isReadOnly).toBe(false);
  });

  it("duplicates via fork and navigates to the new graph", async () => {
    mockGraph = { id: "graph-1", user_id: "other-user" };
    mockLibraryAgent = { id: "lib-1", graph_id: "graph-1" };
    mockForkAgent.mockResolvedValue({
      status: 200,
      data: { id: "lib-2", graph_id: "graph-2" },
    });

    const { result } = renderHook(() => useIsReadOnlyGraph());

    expect(result.current.canDuplicate).toBe(true);

    await act(async () => {
      await result.current.duplicate();
    });

    expect(mockForkAgent).toHaveBeenCalledWith({ libraryAgentId: "lib-1" });
    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith("/build?flowID=graph-2");
    });
  });

  it("shows a toast and does not navigate when duplicate fails", async () => {
    mockGraph = { id: "graph-1", user_id: "other-user" };
    mockLibraryAgent = { id: "lib-1", graph_id: "graph-1" };
    mockForkAgent.mockRejectedValue(new Error("boom"));

    const { result } = renderHook(() => useIsReadOnlyGraph());

    await act(async () => {
      await result.current.duplicate();
    });

    expect(mockPush).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });
});
