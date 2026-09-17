import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getV2GetPendingMessages } from "@/app/api/__generated__/endpoints/chat/chat";
import { useCopilotPendingChips } from "./useCopilotPendingChips";

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  getV2GetPendingMessages: vi.fn(),
}));

type Messages = UIMessage<unknown, UIDataTypes, UITools>[];

const mockGetPending = vi.mocked(getV2GetPendingMessages);

const ASSISTANT_ID = "assistant-stable-id";

function assistantMessage(drainHints = 0): Messages[number] {
  const parts: Messages[number]["parts"] = [
    { type: "text", text: "working…", state: "done" },
  ];
  for (let i = 0; i < drainHints; i++) {
    parts.push({
      type: "data-pending-drained",
      id: `hint-${i}`,
      data: { drainedCount: 1 },
    } as Messages[number]["parts"][number]);
  }
  return { id: ASSISTANT_ID, role: "assistant", parts };
}

// Server reports the buffer as fully drained (count 0) so the hook promotes
// every locally-queued chip to a bubble.
function mockBufferDrained() {
  mockGetPending.mockResolvedValue({
    status: 200,
    data: { count: 0, messages: [] },
    headers: new Headers(),
  } as Awaited<ReturnType<typeof getV2GetPendingMessages>>);
}

function setupHook(initialMessages: Messages) {
  let current: Messages = initialMessages;
  const setMessages = vi.fn(
    (updater: Messages | ((prev: Messages) => Messages)) => {
      current = typeof updater === "function" ? updater(current) : updater;
    },
  );

  const view = renderHook(
    ({ messages }) =>
      useCopilotPendingChips({
        sessionId: "s1",
        status: "streaming",
        messages,
        setMessages,
      }),
    { initialProps: { messages: initialMessages } },
  );

  return { view, setMessages, getMessages: () => current };
}

describe("useCopilotPendingChips", () => {
  beforeEach(() => {
    mockGetPending.mockReset();
    mockBufferDrained();
  });
  afterEach(cleanup);

  it("promotes a queued chip to a bubble the instant a data-pending-drained hint arrives", async () => {
    const { view, getMessages } = setupHook([assistantMessage(0)]);

    act(() => {
      view.result.current.queueMessage("follow up");
    });
    expect(view.result.current.queuedMessages).toEqual(["follow up"]);

    // The backend drains mid-turn and pushes the SSE hint: rerender with the
    // new data-pending-drained part on the streaming assistant message.
    await act(async () => {
      view.rerender({ messages: [assistantMessage(1)] });
    });

    await waitFor(() => {
      expect(mockGetPending).toHaveBeenCalledWith("s1");
      // Chip cleared from the strip…
      expect(view.result.current.queuedMessages).toEqual([]);
    });

    // …and promoted to a user bubble inserted before the streaming assistant.
    const promoted = getMessages().find((m) =>
      m.id.startsWith("promoted-midturn-pending-chip-"),
    );
    expect(promoted?.role).toBe("user");
  });

  it("does not promote when the backend buffer count still covers the local chips", async () => {
    mockGetPending.mockResolvedValue({
      status: 200,
      data: { count: 1, messages: ["follow up"] },
      headers: new Headers(),
    } as Awaited<ReturnType<typeof getV2GetPendingMessages>>);

    const { view, getMessages } = setupHook([assistantMessage(0)]);
    act(() => {
      view.result.current.queueMessage("follow up");
    });

    await act(async () => {
      view.rerender({ messages: [assistantMessage(1)] });
    });

    await waitFor(() => expect(mockGetPending).toHaveBeenCalledWith("s1"));
    expect(
      getMessages().some((m) =>
        m.id.startsWith("promoted-midturn-pending-chip-"),
      ),
    ).toBe(false);
    expect(view.result.current.queuedMessages).toEqual(["follow up"]);
  });

  it("does not promote the previous session's chips after a session switch", async () => {
    let current: Messages = [assistantMessage(0)];
    const setMessages = vi.fn(
      (updater: Messages | ((prev: Messages) => Messages)) => {
        current = typeof updater === "function" ? updater(current) : updater;
      },
    );

    const view = renderHook(
      ({ messages, sessionId }) =>
        useCopilotPendingChips({
          sessionId,
          status: "streaming",
          messages,
          setMessages,
        }),
      {
        initialProps: {
          messages: [assistantMessage(0)],
          sessionId: "s1" as string,
        },
      },
    );

    act(() => {
      view.result.current.queueMessage("old session chip");
    });
    expect(view.result.current.queuedMessages).toEqual(["old session chip"]);

    // Switch to s2 with a HIGHER drain-hint count than s1's baseline while
    // the old chip is still queued for this commit.  The drain effect must
    // re-baseline on the session change and NOT promote the stale chip into
    // the new chat.
    await act(async () => {
      view.rerender({ messages: [assistantMessage(1)], sessionId: "s2" });
    });

    // The session-switch peek rebases the strip to the new (empty) session.
    await waitFor(() => {
      expect(view.result.current.queuedMessages).toEqual([]);
    });

    // No mid-turn promotion of the old chip leaked into the new session.
    expect(
      current.some((m) => m.id.startsWith("promoted-midturn-pending-chip-")),
    ).toBe(false);
  });

  it("backstop poll promotes a chip even without an SSE hint", async () => {
    vi.useFakeTimers();
    try {
      const { view, getMessages } = setupHook([assistantMessage(0)]);
      act(() => {
        view.result.current.queueMessage("follow up");
      });

      // No hint part is ever added; the slow backstop interval must still
      // reconcile against the drained buffer.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10_000);
      });

      expect(mockGetPending).toHaveBeenCalledWith("s1");
      expect(
        getMessages().some((m) =>
          m.id.startsWith("promoted-midturn-pending-chip-"),
        ),
      ).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });
});
