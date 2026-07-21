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

function resetStore() {
  useCopilotUIStore.setState({
    artifactPanel: {
      isOpen: false,
      activeArtifact: null,
      history: [],
      activeTab: "files",
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

  it("clears artifact preview state when sessionId changes", () => {
    useCopilotUIStore.getState().openArtifact(makeArtifact(A_ID, "a.txt"));
    useCopilotUIStore.getState().openArtifact(makeArtifact(B_ID, "b.txt"));

    const { rerender } = renderHook(
      ({ sessionId }) => useAutoOpenArtifacts({ ...defaultProps, sessionId }),
      { initialProps: { sessionId: "s1" } },
    );

    act(() => {
      rerender({ sessionId: "s2" });
    });

    // Switching sessions clears the preview via activeArtifact; we assert on
    // activeArtifact (not isOpen) because the drawer is gated on it.
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.activeArtifact).toBeNull();
    expect(s.history).toEqual([]);
  });

  it("clears artifact preview on unmount (SECRT-2254)", () => {
    useCopilotUIStore.getState().openArtifact(makeArtifact(A_ID, "a.txt"));
    // openArtifact opens the context region (isOpen: true) and drives the
    // preview drawer via activeArtifact.
    expect(useCopilotUIStore.getState().artifactPanel.activeArtifact?.id).toBe(
      A_ID,
    );

    const { unmount } = renderHook(() => useAutoOpenArtifacts(defaultProps));
    act(() => {
      unmount();
    });

    // Drawer is gated on activeArtifact, so clearing it is what hides the
    // preview after nav-away; we assert on activeArtifact here.
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.activeArtifact).toBeNull();
  });
});

// ── registerArtifactForAutoOpen (store unit tests) ────────────────
//
// Auto-open is disabled — registerArtifactForAutoOpen is only kept to track
// known IDs and upgrade metadata when a richer ref arrives for the same id.

describe("registerArtifactForAutoOpen", () => {
  beforeEach(resetStore);

  it("never auto-opens the panel, regardless of readiness", () => {
    useCopilotUIStore.getState().setAutoOpenReady();
    const ref = makeArtifact(A_ID);
    useCopilotUIStore.getState().registerArtifactForAutoOpen(ref);
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("upgrades activeArtifact metadata when a richer ref arrives for a known ID", () => {
    const weakRef: ArtifactRef = {
      id: A_ID,
      title: "File " + A_ID.slice(0, 8),
      mimeType: null,
      sourceUrl: `/api/proxy/api/workspace/files/${A_ID}/download`,
      origin: "agent",
    };
    // First registration tracks the ID but doesn't open the panel.
    useCopilotUIStore.getState().registerArtifactForAutoOpen(weakRef);
    // Simulate the user explicitly opening this artifact afterwards.
    useCopilotUIStore.getState().openArtifact(weakRef);
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
