import { describe, expect, it, beforeEach, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { useAutoOpenArtifacts } from "../useAutoOpenArtifacts";
import { useCopilotUIStore } from "../../../store";

// Capture the real store actions before any test can replace them.
const realOpenArtifact = useCopilotUIStore.getState().openArtifact;
const realResetArtifactPanel = useCopilotUIStore.getState().resetArtifactPanel;

function resetStore() {
  useCopilotUIStore.setState({
    openArtifact: realOpenArtifact,
    resetArtifactPanel: realResetArtifactPanel,
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
  afterEach(resetStore);

  it("does not auto-open artifacts on initial message load", () => {
    renderHook(() => useAutoOpenArtifacts({ sessionId: "session-1" }));
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("does not auto-open when rerendering within the same session", () => {
    const { rerender } = renderHook(
      ({ sessionId }: { sessionId: string }) =>
        useAutoOpenArtifacts({ sessionId }),
      { initialProps: { sessionId: "session-1" } },
    );

    rerender({ sessionId: "session-1" });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("panel should fully reset when session changes", () => {
    const artifact = {
      id: "file1",
      title: "image.png",
      mimeType: "image/png",
      sourceUrl: "/api/proxy/api/workspace/files/file1/download",
      origin: "agent" as const,
    };
    useCopilotUIStore.getState().openArtifact(artifact);
    useCopilotUIStore.getState().openArtifact({
      ...artifact,
      id: "file2",
      title: "second.png",
      sourceUrl: "/api/proxy/api/workspace/files/file2/download",
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    const { rerender } = renderHook(
      ({ sessionId }: { sessionId: string }) =>
        useAutoOpenArtifacts({ sessionId }),
      { initialProps: { sessionId: "session-1" } },
    );

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    rerender({ sessionId: "session-2" });

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(false);
    expect(s.activeArtifact).toBeNull();
    expect(s.history).toEqual([]);
  });
});
