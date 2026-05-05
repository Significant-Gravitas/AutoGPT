import { act, cleanup, renderHook } from "@testing-library/react";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useCopilotUIStore } from "../../store";
import { useAutoOpenArtifacts } from "./useAutoOpenArtifacts";

type Messages = UIMessage<unknown, UIDataTypes, UITools>[];

const A_ID = "11111111-0000-0000-0000-000000000000";
const B_ID = "22222222-0000-0000-0000-000000000000";
const C_ID = "33333333-0000-0000-0000-000000000000";

function makeArtifact(id: string, title = `${id}.txt`) {
  return {
    id,
    title,
    mimeType: "text/plain",
    sourceUrl: `/api/proxy/api/workspace/files/${id}/download`,
    origin: "agent" as const,
  };
}

function makeAgentMessage(
  id: string,
  artifactIds: string[],
): UIMessage<unknown, UIDataTypes, UITools> {
  return {
    id,
    role: "assistant",
    parts: artifactIds.map((aid) => ({
      type: "file" as const,
      url: `/api/proxy/api/workspace/files/${aid}/download`,
      filename: `${aid}.txt`,
      mediaType: "text/plain",
    })),
  };
}

function makeUserMessage(
  id: string,
  artifactIds: string[],
): UIMessage<unknown, UIDataTypes, UITools> {
  return {
    id,
    role: "user",
    parts: artifactIds.map((aid) => ({
      type: "file" as const,
      url: `/api/proxy/api/workspace/files/${aid}/download`,
      filename: `${aid}.txt`,
      mediaType: "text/plain",
    })),
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

const defaultProps = {
  sessionId: "s1",
  messages: [] as Messages,
  isLoadingSession: false,
  isArtifactsEnabled: true,
};

describe("useAutoOpenArtifacts", () => {
  beforeEach(resetStore);
  afterEach(cleanup);

  it("does not auto-open on initial render", () => {
    renderHook(() => useAutoOpenArtifacts(defaultProps));
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("does not auto-open when rerendering within the same session", () => {
    const { rerender } = renderHook(
      ({ sessionId }) => useAutoOpenArtifacts({ ...defaultProps, sessionId }),
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
      ({ sessionId }) => useAutoOpenArtifacts({ ...defaultProps, sessionId }),
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
      ({ sessionId }) => useAutoOpenArtifacts({ ...defaultProps, sessionId }),
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

  it("closes the panel on unmount so nav-away → nav-back doesn't resurrect it (SECRT-2254)", () => {
    useCopilotUIStore.getState().openArtifact(makeArtifact(A_ID, "a.txt"));
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    const { unmount } = renderHook(() => useAutoOpenArtifacts(defaultProps));

    act(() => {
      unmount();
    });

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(false);
    expect(s.activeArtifact).toBeNull();
    expect(s.history).toEqual([]);
  });

  it("does not re-open a panel whose store state is stale on fresh mount (SECRT-2220)", () => {
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

    const { unmount } = renderHook(() => useAutoOpenArtifacts(defaultProps));
    act(() => {
      unmount();
    });

    renderHook(() => useAutoOpenArtifacts(defaultProps));
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("auto-opens when a new agent artifact appears after the initial snapshot", () => {
    const { rerender } = renderHook(
      ({ messages }) => useAutoOpenArtifacts({ ...defaultProps, messages }),
      { initialProps: { messages: [] as Messages } },
    );

    act(() => {
      rerender({ messages: [makeAgentMessage("m1", [A_ID])] });
    });

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(true);
    expect(s.activeArtifact?.id).toBe(A_ID);
  });

  it("does not auto-open artifacts already present at session load", () => {
    renderHook(() =>
      useAutoOpenArtifacts({
        ...defaultProps,
        messages: [makeAgentMessage("m1", [A_ID])],
      }),
    );

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("does not auto-open user-uploaded artifacts", () => {
    const { rerender } = renderHook(
      ({ messages }) => useAutoOpenArtifacts({ ...defaultProps, messages }),
      { initialProps: { messages: [] as Messages } },
    );

    act(() => {
      rerender({ messages: [makeUserMessage("m1", [A_ID])] });
    });

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("opens the most recent artifact when multiple new ones appear simultaneously", () => {
    const { rerender } = renderHook(
      ({ messages }) => useAutoOpenArtifacts({ ...defaultProps, messages }),
      { initialProps: { messages: [] as Messages } },
    );

    act(() => {
      rerender({
        messages: [makeAgentMessage("m1", [A_ID, B_ID, C_ID])],
      });
    });

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(true);
    expect(s.activeArtifact?.id).toBe(C_ID);
  });

  it("does not auto-open after the user explicitly closes the panel", () => {
    const { rerender } = renderHook(
      ({ messages }) => useAutoOpenArtifacts({ ...defaultProps, messages }),
      { initialProps: { messages: [] as Messages } },
    );

    act(() => {
      rerender({ messages: [makeAgentMessage("m1", [A_ID])] });
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    act(() => {
      useCopilotUIStore.getState().closeArtifactPanel();
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);

    act(() => {
      rerender({
        messages: [
          makeAgentMessage("m1", [A_ID]),
          makeAgentMessage("m2", [B_ID]),
        ],
      });
    });

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("does not re-trigger auto-open when isLoadingSession pulses (reconnect)", () => {
    const { rerender } = renderHook(
      ({ messages, isLoadingSession }) =>
        useAutoOpenArtifacts({
          ...defaultProps,
          messages,
          isLoadingSession,
        }),
      {
        initialProps: {
          messages: [makeAgentMessage("m1", [A_ID])] as Messages,
          isLoadingSession: false,
        },
      },
    );

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);

    act(() => {
      rerender({
        messages: [makeAgentMessage("m1", [A_ID])],
        isLoadingSession: true,
      });
    });
    act(() => {
      rerender({
        messages: [makeAgentMessage("m1", [A_ID])],
        isLoadingSession: false,
      });
    });

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("resets close-suppression when session changes (remount)", () => {
    const { rerender, unmount } = renderHook(
      ({ sessionId, messages }) =>
        useAutoOpenArtifacts({ ...defaultProps, sessionId, messages }),
      { initialProps: { sessionId: "s1", messages: [] as Messages } },
    );

    // New artifact → opens
    act(() => {
      rerender({
        sessionId: "s1",
        messages: [makeAgentMessage("m1", [A_ID])],
      });
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    // User closes
    act(() => {
      useCopilotUIStore.getState().closeArtifactPanel();
    });

    // Unmount (simulates key={sessionId} remount)
    act(() => {
      unmount();
    });
    resetStore();

    // Remount with new session — close suppression should be cleared
    const { rerender: rerender2 } = renderHook(
      ({ messages }) =>
        useAutoOpenArtifacts({
          ...defaultProps,
          sessionId: "s2",
          messages,
        }),
      { initialProps: { messages: [] as Messages } },
    );

    act(() => {
      rerender2({ messages: [makeAgentMessage("m2", [B_ID])] });
    });

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);
  });

  it("does not auto-open when sessionId is null", () => {
    renderHook(() =>
      useAutoOpenArtifacts({
        ...defaultProps,
        sessionId: null,
        messages: [makeAgentMessage("m1", [A_ID])],
      }),
    );

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("does not auto-open when artifacts feature flag is disabled", () => {
    const { rerender } = renderHook(
      ({ messages }) =>
        useAutoOpenArtifacts({
          ...defaultProps,
          isArtifactsEnabled: false,
          messages,
        }),
      { initialProps: { messages: [] as Messages } },
    );

    act(() => {
      rerender({ messages: [makeAgentMessage("m1", [A_ID])] });
    });

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });
});
