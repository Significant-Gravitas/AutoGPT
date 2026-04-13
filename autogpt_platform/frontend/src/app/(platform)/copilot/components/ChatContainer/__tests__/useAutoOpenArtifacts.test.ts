import { describe, expect, it, beforeEach, afterEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { useAutoOpenArtifacts } from "../useAutoOpenArtifacts";
import { useCopilotUIStore } from "../../../store";
import type { UIMessage, UIDataTypes, UITools } from "ai";

function makeMessage(
  id: string,
  role: "assistant" | "user",
  parts: UIMessage<unknown, UIDataTypes, UITools>["parts"] = [],
): UIMessage<unknown, UIDataTypes, UITools> {
  return {
    id,
    role,
    parts,
    createdAt: new Date(),
    content: "",
  } as UIMessage<unknown, UIDataTypes, UITools>;
}

// Capture the real store actions before any test can replace them.
const realOpenArtifact = useCopilotUIStore.getState().openArtifact;
const realCloseArtifactPanel = useCopilotUIStore.getState().closeArtifactPanel;

function resetStore() {
  useCopilotUIStore.setState({
    openArtifact: realOpenArtifact,
    closeArtifactPanel: realCloseArtifactPanel,
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
    const messages = [
      makeMessage("m1", "user"),
      makeMessage("m2", "assistant", [
        {
          type: "text" as const,
          text: "Here is workspace://file1#image/png",
        },
      ]),
    ];

    renderHook(() =>
      useAutoOpenArtifacts({ messages, sessionId: "session-1" }),
    );

    // First render is initialization — should NOT auto-open existing artifacts
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("resets hydration baseline when sessionId changes", () => {
    const messages = [
      makeMessage("m1", "assistant", [
        {
          type: "text" as const,
          text: "Here is workspace://file1#image/png",
        },
      ]),
    ];

    const { rerender } = renderHook(
      ({ sessionId }: { sessionId: string }) =>
        useAutoOpenArtifacts({ messages, sessionId }),
      { initialProps: { sessionId: "session-1" } },
    );

    // Switch session — should not auto-open existing artifacts
    rerender({ sessionId: "session-2" });

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("panel should close when session changes", () => {
    // Pre-open the panel as if an artifact was viewed in session-1
    const artifact = {
      id: "file1",
      title: "image.png",
      mimeType: "image/png",
      sourceUrl: "/api/proxy/api/workspace/files/file1/download",
      origin: "agent" as const,
    };
    useCopilotUIStore.getState().openArtifact(artifact);
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    const messages = [
      makeMessage("m1", "assistant", [
        {
          type: "text" as const,
          text: "Here is workspace://file1#image/png",
        },
      ]),
    ];

    const { rerender } = renderHook(
      ({ sessionId }: { sessionId: string }) =>
        useAutoOpenArtifacts({ messages, sessionId }),
      { initialProps: { sessionId: "session-1" } },
    );

    // Panel should still be open after initial render
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    // Switch to a different session
    rerender({ sessionId: "session-2" });

    // Panel should be closed after session switch
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });
});
