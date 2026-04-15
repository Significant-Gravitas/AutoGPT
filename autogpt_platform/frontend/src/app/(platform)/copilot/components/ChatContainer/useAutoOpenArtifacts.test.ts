import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
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
});
