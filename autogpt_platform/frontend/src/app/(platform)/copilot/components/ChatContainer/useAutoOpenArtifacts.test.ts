import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useCopilotUIStore } from "../../store";
import { useAutoOpenArtifacts } from "./useAutoOpenArtifacts";

function assistantMessageWithText(id: string, text: string) {
  return {
    id,
    role: "assistant" as const,
    parts: [{ type: "text" as const, text }],
  };
}

const A_ID = "11111111-0000-0000-0000-000000000000";
const B_ID = "22222222-0000-0000-0000-000000000000";

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

  it("does NOT auto-open on the initial hydration of message list (baseline pass)", () => {
    const messages = [
      assistantMessageWithText("m1", `[a](workspace://${A_ID})`),
    ];
    renderHook(() =>
      useAutoOpenArtifacts({ messages: messages as any, sessionId: "s1" }),
    );
    // Initial run just records the baseline fingerprint; nothing opens.
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("auto-opens when an existing assistant message adds a new artifact", () => {
    // 1st render: baseline with no artifact.
    const initial = [assistantMessageWithText("m1", "thinking...")];
    const { rerender } = renderHook(
      ({ messages, sessionId }) =>
        useAutoOpenArtifacts({ messages: messages as any, sessionId }),
      { initialProps: { messages: initial, sessionId: "s1" } },
    );
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);

    // 2nd render: same message id now contains an artifact link.
    act(() => {
      rerender({
        messages: [
          assistantMessageWithText("m1", `here: [A](workspace://${A_ID})`),
        ],
        sessionId: "s1",
      });
    });
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(true);
    expect(s.activeArtifact?.id).toBe(A_ID);
  });

  it("does not re-open when the fingerprint hasn't changed", () => {
    const msg = assistantMessageWithText("m1", `[A](workspace://${A_ID})`);
    const { rerender } = renderHook(
      ({ messages, sessionId }) =>
        useAutoOpenArtifacts({ messages: messages as any, sessionId }),
      { initialProps: { messages: [msg], sessionId: "s1" } },
    );
    // Baseline captured; no open.
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);

    // Rerender identical content: no change in fingerprint → no open.
    act(() => {
      rerender({ messages: [msg], sessionId: "s1" });
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("auto-opens when a brand-new assistant message arrives after the baseline is established", () => {
    // First render: one message without artifacts → establishes baseline.
    const { rerender } = renderHook(
      ({ messages, sessionId }) =>
        useAutoOpenArtifacts({ messages: messages as any, sessionId }),
      {
        initialProps: {
          messages: [assistantMessageWithText("m1", "plain")] as any,
          sessionId: "s1",
        },
      },
    );
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);

    // Second render: a *new* assistant message with an artifact. Baseline
    // is already set, so this should auto-open.
    act(() => {
      rerender({
        messages: [
          assistantMessageWithText("m1", "plain"),
          assistantMessageWithText("m2", `[B](workspace://${B_ID})`),
        ] as any,
        sessionId: "s1",
      });
    });
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(true);
    expect(s.activeArtifact?.id).toBe(B_ID);
  });

  it("resets hydration baseline when sessionId changes", () => {
    const { rerender } = renderHook(
      ({ messages, sessionId }) =>
        useAutoOpenArtifacts({ messages: messages as any, sessionId }),
      {
        initialProps: {
          messages: [
            assistantMessageWithText("m1", `[A](workspace://${A_ID})`),
          ] as any,
          sessionId: "s1",
        },
      },
    );
    // Switch to a new session — the first pass on the new session should
    // NOT auto-open (it's a fresh hydration).
    act(() => {
      rerender({
        messages: [
          assistantMessageWithText("m2", `[B](workspace://${B_ID})`),
        ] as any,
        sessionId: "s2",
      });
    });
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });
});
