import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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

import { useIsReadOnlyGraph } from "../hooks/useIsReadOnlyGraph";

describe("useIsReadOnlyGraph", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUser = { id: "user-1" };
    mockGraph = { id: "graph-1", user_id: "user-1" };
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

  it("is not read-only while the graph is still loading", () => {
    mockGraph = undefined;

    const { result } = renderHook(() => useIsReadOnlyGraph());

    expect(result.current.isReadOnly).toBe(false);
  });

  it("is not read-only while the user is still loading", () => {
    mockUser = null;
    mockGraph = { id: "graph-1", user_id: "other-user" };

    const { result } = renderHook(() => useIsReadOnlyGraph());

    expect(result.current.isReadOnly).toBe(false);
  });
});
