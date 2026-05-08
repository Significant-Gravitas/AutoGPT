import { act, cleanup, renderHook } from "@testing-library/react";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ArtifactRef } from "../../store";
import { useCopilotUIStore } from "../../store";
import { useAutoOpenArtifacts } from "./useAutoOpenArtifacts";

type Messages = UIMessage<unknown, UIDataTypes, UITools>[];

const A_ID = "11111111-0000-0000-0000-000000000000";
const B_ID = "22222222-0000-0000-0000-000000000000";

function makeArtifact(
  id: string,
  title = `${id}.txt`,
  origin: ArtifactRef["origin"] = "agent",
): ArtifactRef {
  return {
    id,
    title,
    mimeType: "text/plain",
    sourceUrl: `/api/proxy/api/workspace/files/${id}/download`,
    origin,
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
  useCopilotUIStore.getState().resetAutoOpenState();
}

const defaultProps = {
  sessionId: "s1",
  messages: [] as Messages,
  isLoadingSession: false,
  isArtifactsEnabled: true,
};

describe("useAutoOpenArtifacts (card-based)", () => {
  beforeEach(resetStore);
  afterEach(cleanup);

  // ── Lifecycle ──────────────────────────────────────────────────────

  it("does not open the panel on initial render", () => {
    renderHook(() => useAutoOpenArtifacts(defaultProps));
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

  it("closes the panel on unmount (SECRT-2254)", () => {
    useCopilotUIStore.getState().openArtifact(makeArtifact(A_ID, "a.txt"));
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    const { unmount } = renderHook(() => useAutoOpenArtifacts(defaultProps));
    act(() => {
      unmount();
    });

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(false);
    expect(s.activeArtifact).toBeNull();
  });

  // ── Readiness gating ──────────────────────────────────────────────

  it("sets auto-open ready after loading completes with a session", () => {
    renderHook(() => useAutoOpenArtifacts(defaultProps));
    // After the effect fires, registerArtifactForAutoOpen should auto-open
    const ref = makeArtifact(A_ID);
    act(() => {
      useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);
  });

  it("does not set ready when sessionId is null", () => {
    renderHook(() =>
      useAutoOpenArtifacts({ ...defaultProps, sessionId: null }),
    );
    const ref = makeArtifact(A_ID);
    act(() => {
      useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("does not set ready when artifacts feature flag is disabled", () => {
    renderHook(() =>
      useAutoOpenArtifacts({ ...defaultProps, isArtifactsEnabled: false }),
    );
    const ref = makeArtifact(A_ID);
    act(() => {
      useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("defers readiness when session was loading and messages haven't hydrated", () => {
    const { rerender } = renderHook(
      ({ isLoadingSession, messages }) =>
        useAutoOpenArtifacts({
          ...defaultProps,
          isLoadingSession,
          messages,
        }),
      { initialProps: { isLoadingSession: true, messages: [] as Messages } },
    );

    // Loading done but messages still empty → should NOT be ready
    act(() => {
      rerender({ isLoadingSession: false, messages: [] as Messages });
    });
    const ref = makeArtifact(A_ID);
    act(() => {
      useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);

    // Once messages arrive → ready
    resetStore();
    act(() => {
      rerender({
        isLoadingSession: false,
        messages: [makeAgentMessage("m1", [A_ID])],
      });
    });
    const ref2 = makeArtifact(B_ID);
    act(() => {
      useCopilotUIStore.getState().registerArtifactForAutoOpen(ref2);
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);
  });

  // ── User-close suppression ────────────────────────────────────────

  it("suppresses auto-open after the user explicitly closes the panel", () => {
    renderHook(() => useAutoOpenArtifacts(defaultProps));

    // Open via store
    act(() => {
      useCopilotUIStore.getState().openArtifact(makeArtifact(A_ID));
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    // User closes
    act(() => {
      useCopilotUIStore.getState().closeArtifactPanel();
    });

    // New registration should NOT auto-open
    const ref = makeArtifact(B_ID);
    act(() => {
      useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("resets close-suppression when session changes (remount)", () => {
    renderHook(() => useAutoOpenArtifacts(defaultProps));

    // Open + close → suppressed
    act(() => {
      useCopilotUIStore.getState().openArtifact(makeArtifact(A_ID));
    });
    act(() => {
      useCopilotUIStore.getState().closeArtifactPanel();
    });

    // Session change resets suppression
    const { rerender } = renderHook(
      ({ sessionId }) => useAutoOpenArtifacts({ ...defaultProps, sessionId }),
      { initialProps: { sessionId: "s1" } },
    );
    act(() => {
      rerender({ sessionId: "s2" });
    });

    // New registration should auto-open
    const ref = makeArtifact(B_ID);
    act(() => {
      useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);
  });
});

// ── registerArtifactForAutoOpen (store unit tests) ────────────────

describe("registerArtifactForAutoOpen", () => {
  beforeEach(resetStore);

  it("does not auto-open when not ready", () => {
    const ref = makeArtifact(A_ID);
    useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("auto-opens an agent artifact when ready", () => {
    useCopilotUIStore.getState().setAutoOpenReady();
    const ref = makeArtifact(A_ID);
    useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);
    expect(useCopilotUIStore.getState().artifactPanel.activeArtifact?.id).toBe(
      A_ID,
    );
  });

  it("does not auto-open user-uploaded artifacts", () => {
    useCopilotUIStore.getState().setAutoOpenReady();
    const ref = makeArtifact(A_ID, "upload.txt", "user-upload");
    useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("does not auto-open the same artifact twice", () => {
    useCopilotUIStore.getState().setAutoOpenReady();
    const ref = makeArtifact(A_ID);
    useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    // Close, then re-register same ID
    useCopilotUIStore.getState().closeArtifactPanel();
    // Reset the user-closed flag so it doesn't interfere
    useCopilotUIStore.getState().resetAutoOpenState();
    useCopilotUIStore.getState().setAutoOpenReady();
    // But A_ID is already known from the first registration — wait,
    // resetAutoOpenState clears known IDs too. Let's re-add it as known.
    useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    // First call after reset → opens again (this is correct — reset clears known)
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    // NOW close and try again without reset — same ID, should not re-trigger
    useCopilotUIStore.getState().closeArtifactPanel();
    // Manually set ready again (close doesn't affect ready)
    const ref2 = makeArtifact(A_ID);
    useCopilotUIStore.getState().registerArtifactForAutoOpen(ref2);
    // Already known, won't re-open
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("does not auto-open when user-closed flag is set", () => {
    useCopilotUIStore.getState().setAutoOpenReady();
    useCopilotUIStore.getState().markUserClosedForAutoOpen();
    const ref = makeArtifact(A_ID);
    useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("upgrades activeArtifact metadata when a richer ref arrives for a known ID", () => {
    useCopilotUIStore.getState().setAutoOpenReady();
    const weakRef: ArtifactRef = {
      id: A_ID,
      title: "File " + A_ID.slice(0, 8),
      mimeType: null,
      sourceUrl: `/api/proxy/api/workspace/files/${A_ID}/download`,
      origin: "agent",
    };
    useCopilotUIStore.getState().registerArtifactForAutoOpen(weakRef);
    expect(
      useCopilotUIStore.getState().artifactPanel.activeArtifact?.mimeType,
    ).toBeNull();

    const richRef: ArtifactRef = {
      id: A_ID,
      title: "plan.md",
      mimeType: "text/markdown",
      sourceUrl: `/api/proxy/api/workspace/files/${A_ID}/download`,
      origin: "agent",
    };
    useCopilotUIStore.getState().registerArtifactForAutoOpen(richRef);
    expect(
      useCopilotUIStore.getState().artifactPanel.activeArtifact?.mimeType,
    ).toBe("text/markdown");
  });
});
