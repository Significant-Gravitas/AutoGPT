import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockSaveGraph = vi.fn();
vi.mock("@/app/(platform)/build/hooks/useSaveGraph", () => ({
  useSaveGraph: () => ({ saveGraph: mockSaveGraph, isSaving: false }),
}));

const graphState = {
  hasInputs: () => false,
  hasCredentials: () => false,
};
vi.mock("@/app/(platform)/build/stores/graphStore", () => ({
  useGraphStore: (selector: (s: typeof graphState) => unknown) =>
    selector(graphState),
}));

import { useScheduleGraph } from "./useScheduleGraph";

describe("useScheduleGraph", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("opens the scheduler once the graph is saved", async () => {
    mockSaveGraph.mockResolvedValue({ id: "graph-1", version: 2 });

    const { result } = renderHook(() => useScheduleGraph());

    await act(async () => {
      await result.current.handleScheduleGraph();
    });

    expect(result.current.openCronSchedulerDialog).toBe(true);
  });

  it("schedules nothing when the save is refused", async () => {
    // A refused save resolves to null after telling the user why; scheduling
    // would schedule the graph from before their edits.
    mockSaveGraph.mockResolvedValue(null);

    const { result } = renderHook(() => useScheduleGraph());

    await act(async () => {
      await result.current.handleScheduleGraph();
    });

    expect(result.current.openCronSchedulerDialog).toBe(false);
    expect(result.current.openScheduleInputDialog).toBe(false);
  });
});
