import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useCopilotUIStore } from "../../store";
import { useAutoOpenArtifacts } from "./useAutoOpenArtifacts";

const A_ID = "11111111-0000-0000-0000-000000000000";
const B_ID = "22222222-0000-0000-0000-000000000000";

function makeArtifact(id: string, title = `${id}.txt`) {
  return {
    id,
    title,
    mimeType: "text/plain",
    sourceUrl: `/api/proxy/api/workspace/files/${id}/download`,
    origin: "agent" as const,
  };
}

function resetStore() {
  useCopilotUIStore.setState({
    artifactPanel: {
      isOpen: false,
      isMinimized: false,
      isMaximized: false,
      width: 600,
      activeArtifact: null,
      history: [],
    },
  });
}

describe("useAutoOpenArtifacts", () => {
  beforeEach(resetStore);
  // Testing Library auto-cleanup isn't registered in our Vitest setup, so
  // mounted `renderHook` instances (and their unmount cleanups) would leak
  // between tests — here the unmount effect in useAutoOpenArtifacts would
  // fire after the next test had already run and corrupt its assertions.
  afterEach(cleanup);

  it("does not auto-open on initial render", () => {
    renderHook(() => useAutoOpenArtifacts({ sessionId: "s1" }));
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("does not auto-open when rerendering within the same session", () => {
    const { rerender } = renderHook(
      ({ sessionId }) => useAutoOpenArtifacts({ sessionId }),
      { initialProps: { sessionId: "s1" } },
    );

    act(() => {
      rerender({ sessionId: "s1" });
    });

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("resets the panel state when sessionId changes", () => {
    useCopilotUIStore.getState().openArtifact(makeArtifact(A_ID, "a.txt"));
    useCopilotUIStore.getState().openArtifact(makeArtifact(B_ID, "b.txt"));

    const { rerender } = renderHook(
      ({ sessionId }) => useAutoOpenArtifacts({ sessionId }),
      { initialProps: { sessionId: "s1" } },
    );

    act(() => {
      rerender({ sessionId: "s2" });
    });

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(false);
    expect(s.activeArtifact).toBeNull();
    expect(s.history).toEqual([]);
  });

  it("does not carry a stale back stack into the next session", () => {
    useCopilotUIStore.getState().openArtifact(makeArtifact(A_ID, "a.txt"));
    useCopilotUIStore.getState().openArtifact(makeArtifact(B_ID, "b.txt"));

    const { rerender } = renderHook(
      ({ sessionId }) => useAutoOpenArtifacts({ sessionId }),
      { initialProps: { sessionId: "s1" } },
    );

    act(() => {
      rerender({ sessionId: "s2" });
    });

    useCopilotUIStore.getState().openArtifact(makeArtifact("c", "c.txt"));

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.activeArtifact?.id).toBe("c");
    expect(s.history).toEqual([]);
  });

  // SECRT-2254: "had agent panel open then went to profile then went to home
  // and agent panel was still open". Nav-away unmounts the copilot page; if
  // the panel state persists in the store, coming back re-renders it open.
  it("closes the panel on unmount so nav-away → nav-back doesn't resurrect it (SECRT-2254)", () => {
    useCopilotUIStore.getState().openArtifact(makeArtifact(A_ID, "a.txt"));
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    const { unmount } = renderHook(() =>
      useAutoOpenArtifacts({ sessionId: "s1" }),
    );

    act(() => {
      unmount();
    });

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(false);
    expect(s.activeArtifact).toBeNull();
    expect(s.history).toEqual([]);
  });

  // SECRT-2220: "keep closed by default" — a fresh mount (e.g. user returns to
  // /copilot) must start with a closed panel even if the store somehow carries
  // stale state from a prior life.
  it("does not re-open a panel whose store state is stale on fresh mount (SECRT-2220)", () => {
    // Simulate the store being left in an open state by a previous page life.
    useCopilotUIStore.setState({
      artifactPanel: {
        isOpen: true,
        isMinimized: false,
        isMaximized: false,
        width: 600,
        activeArtifact: makeArtifact(A_ID, "stale.txt"),
        history: [],
      },
    });

    const { unmount } = renderHook(() =>
      useAutoOpenArtifacts({ sessionId: "s1" }),
    );
    act(() => {
      unmount();
    });

    // Next mount of the page should see a clean store.
    renderHook(() => useAutoOpenArtifacts({ sessionId: "s1" }));
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });
});
