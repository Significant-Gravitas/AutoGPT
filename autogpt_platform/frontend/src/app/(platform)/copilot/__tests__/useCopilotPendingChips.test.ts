import { act, renderHook, waitFor } from "@testing-library/react";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// ── Mock the generated API endpoint so we can control peek responses. ──
const peekMock = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  getV2GetPendingMessages: (...args: unknown[]) => peekMock(...args),
}));

// Import AFTER mocks are declared.
import { useCopilotPendingChips } from "../useCopilotPendingChips";

function user(id: string, text?: string): UIMessage {
  return {
    id,
    role: "user",
    parts: [
      {
        type: "text" as const,
        text: text ?? `user-${id}`,
        state: "done" as const,
      },
    ],
  };
}

function assistant(id: string, text?: string): UIMessage {
  return {
    id,
    role: "assistant",
    parts: [
      {
        type: "text" as const,
        text: text ?? `assistant-${id}`,
        state: "done" as const,
      },
    ],
  };
}

describe("useCopilotPendingChips", () => {
  beforeEach(() => {
    peekMock.mockReset();
    peekMock.mockResolvedValue({
      status: 200,
      data: { count: 0, messages: [] },
    });
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("restores chips from backend peek on session load", async () => {
    peekMock.mockResolvedValueOnce({
      status: 200,
      data: { count: 2, messages: ["first", "second"] },
    });

    const setMessages = vi.fn();
    const { result } = renderHook(() =>
      useCopilotPendingChips({
        sessionId: "sess-1",
        status: "ready",
        messages: [],
        setMessages,
      }),
    );

    await waitFor(() => {
      expect(result.current.queuedMessages).toEqual(["first", "second"]);
    });
  });

  it("clears chips when the session changes", async () => {
    // Initial session returns 2 chips.
    peekMock.mockResolvedValueOnce({
      status: 200,
      data: { count: 2, messages: ["a", "b"] },
    });
    const setMessages = vi.fn();
    const { result, rerender } = renderHook(
      ({ sessionId }: { sessionId: string | null }) =>
        useCopilotPendingChips({
          sessionId,
          status: "ready",
          messages: [],
          setMessages,
        }),
      { initialProps: { sessionId: "sess-1" } },
    );

    await waitFor(() => {
      expect(result.current.queuedMessages).toHaveLength(2);
    });

    // Next session's peek returns no chips.
    peekMock.mockResolvedValueOnce({
      status: 200,
      data: { count: 0, messages: [] },
    });
    rerender({ sessionId: "sess-2" });

    await waitFor(() => {
      expect(result.current.queuedMessages).toEqual([]);
    });
  });

  it("appendChip adds a chip locally without hitting backend", () => {
    const setMessages = vi.fn();
    const { result } = renderHook(() =>
      useCopilotPendingChips({
        sessionId: "s",
        status: "ready",
        messages: [],
        setMessages,
      }),
    );

    act(() => {
      result.current.appendChip("hello");
    });
    expect(result.current.queuedMessages).toEqual(["hello"]);
  });

  it("promotes chips to a bubble when a second new assistant appears (auto-continue)", () => {
    const setMessages = vi.fn();

    // Initial: only a user msg (no assistants yet). The hook's internal
    // "seen" set starts empty so the FIRST new assistant in a streaming
    // chain is treated as Turn 1's opener.
    const { result, rerender } = renderHook(
      ({ messages }: { messages: UIMessage[] }) =>
        useCopilotPendingChips({
          sessionId: "s",
          status: "streaming",
          messages,
          setMessages,
        }),
      { initialProps: { messages: [user("u1")] } },
    );

    // First assistant appears → opener; no promotion (correctly).
    rerender({
      messages: [user("u1"), assistant("a1", "turn1")],
    });

    // User queues a chip while turn 1 is still streaming.
    act(() => {
      result.current.appendChip("followup");
    });

    // Clear any prior setMessages calls so we only inspect the promotion.
    setMessages.mockClear();

    // A SECOND new assistant id appears → auto-continue detected.
    rerender({
      messages: [
        user("u1"),
        assistant("a1", "turn1"),
        assistant("a2", "turn2"),
      ],
    });

    // setMessages was called with an updater that inserts the promoted bubble.
    const promoteCall = setMessages.mock.calls.find(
      ([fn]) => typeof fn === "function",
    );
    expect(promoteCall).toBeDefined();

    const updater = promoteCall![0] as (prev: UIMessage[]) => UIMessage[];
    const after = updater([user("u1"), assistant("a1"), assistant("a2")]);

    const a2Idx = after.findIndex((m) => m.id === "a2");
    const promotedIdx = after.findIndex((m) =>
      m.id.startsWith("promoted-auto-continue-"),
    );
    expect(promotedIdx).toBeGreaterThanOrEqual(0);
    expect(promotedIdx).toBeLessThan(a2Idx);
    expect(after[promotedIdx].role).toBe("user");
    expect((after[promotedIdx].parts![0] as { text?: string }).text).toBe(
      "followup",
    );

    // And chips have been cleared.
    expect(result.current.queuedMessages).toEqual([]);
  });

  it("turn-start drain: peek count=0 clears chips once submitted→streaming", async () => {
    const setMessages = vi.fn();
    // Seed with chips via the idle-state peek so local state has them by
    // the time we kick the submitted→streaming transition.
    peekMock.mockResolvedValueOnce({
      status: 200,
      data: { count: 1, messages: ["local-chip"] },
    });

    type StatusProps = {
      status: "ready" | "submitted" | "streaming";
    };
    const { result, rerender } = renderHook<
      ReturnType<typeof useCopilotPendingChips>,
      StatusProps
    >(
      ({ status }) =>
        useCopilotPendingChips({
          sessionId: "s",
          status,
          messages: [],
          setMessages,
        }),
      { initialProps: { status: "ready" } },
    );

    await waitFor(() => {
      expect(result.current.queuedMessages).toEqual(["local-chip"]);
    });

    rerender({ status: "submitted" });
    // Backend really drained — count is 0.
    peekMock.mockResolvedValue({
      status: 200,
      data: { count: 0, messages: [] },
    });
    rerender({ status: "streaming" });

    await waitFor(() => {
      expect(result.current.queuedMessages).toEqual([]);
    });
  });

  it("turn-start drain: non-200 peek response does not clear chips", async () => {
    const setMessages = vi.fn();
    // Seed chips from the session-load peek so we don't race with a
    // separate appendChip after idle-peek has already overwritten state.
    peekMock.mockResolvedValueOnce({
      status: 200,
      data: { count: 1, messages: ["keep-me"] },
    });

    type StatusProps = {
      status: "ready" | "submitted" | "streaming";
    };
    const { result, rerender } = renderHook<
      ReturnType<typeof useCopilotPendingChips>,
      StatusProps
    >(
      ({ status }) =>
        useCopilotPendingChips({
          sessionId: "s",
          status,
          messages: [],
          setMessages,
        }),
      { initialProps: { status: "ready" } },
    );

    await waitFor(() => {
      expect(result.current.queuedMessages).toEqual(["keep-me"]);
    });

    rerender({ status: "submitted" });
    // Turn-start peek returns an error status — the condition `count === 0`
    // only fires on `status === 200`, so chips must stay.
    peekMock.mockResolvedValue({ status: 500, data: undefined });
    rerender({ status: "streaming" });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(result.current.queuedMessages).toEqual(["keep-me"]);
  });

  it("mid-turn poll: peek error is swallowed and chips are preserved", async () => {
    vi.useFakeTimers();
    const setMessages = vi.fn();
    const { result } = renderHook(() =>
      useCopilotPendingChips({
        sessionId: "s",
        status: "streaming",
        messages: [user("u1"), assistant("a1")],
        setMessages,
      }),
    );

    act(() => {
      result.current.appendChip("survives");
    });

    // Simulate a transient network failure on the peek.
    peekMock.mockRejectedValue(new Error("network blip"));

    await act(async () => {
      vi.advanceTimersByTime(2_000);
      await Promise.resolve();
      await Promise.resolve();
    });

    // No promotion happened, chips remain intact.
    expect(result.current.queuedMessages).toEqual(["survives"]);
    const promoteCall = setMessages.mock.calls.find(([arg]) => {
      if (typeof arg !== "function") return false;
      const updated = (arg as (p: UIMessage[]) => UIMessage[])([]);
      return updated.some((m) => m.id.startsWith("promoted-"));
    });
    expect(promoteCall).toBeUndefined();
  });

  it("mid-turn poll promotes drained chips when backend count drops", async () => {
    vi.useFakeTimers();
    const setMessages = vi.fn();
    const { result } = renderHook(() =>
      useCopilotPendingChips({
        sessionId: "s",
        status: "streaming",
        messages: [user("u1"), assistant("a1")],
        setMessages,
      }),
    );

    act(() => {
      result.current.appendChip("chipA");
      result.current.appendChip("chipB");
    });

    // Backend now reports count=0 (drained by MCP wrapper).
    peekMock.mockResolvedValue({
      status: 200,
      data: { count: 0, messages: [] },
    });

    await act(async () => {
      vi.advanceTimersByTime(2_000);
      // Let the awaited promise resolve.
      await Promise.resolve();
      await Promise.resolve();
    });

    // The poll should have promoted the two chips to a bubble.
    const promotedCall = setMessages.mock.calls.find(([arg]) => {
      if (typeof arg !== "function") return false;
      const updated = (arg as (p: UIMessage[]) => UIMessage[])([]);
      return updated.some((m) => m.id.startsWith("promoted-midturn-"));
    });
    expect(promotedCall).toBeDefined();

    // And — crucially — the promoted bubble is inserted BEFORE the
    // trailing streaming assistant, not after.  The AI SDK's ``useChat``
    // streams SSE deltas into ``messages[-1]``; if the last message is
    // a user bubble instead of the still-streaming assistant, every
    // subsequent chunk lands in the wrong slot and the UI freezes until
    // a page refresh.
    const streamingUpdater = promotedCall![0] as (
      prev: UIMessage[],
    ) => UIMessage[];
    const priorMessages = [user("u1"), assistant("a1", "streaming...")];
    const afterPromotion = streamingUpdater(priorMessages);
    expect(afterPromotion).toHaveLength(3);
    const lastIdx = afterPromotion.length - 1;
    expect(afterPromotion[lastIdx].role).toBe("assistant");
    expect(afterPromotion[lastIdx].id).toBe("a1");
    expect(afterPromotion[lastIdx - 1].id.startsWith("promoted-midturn-")).toBe(
      true,
    );
    // And remaining chips cleared.
    expect(result.current.queuedMessages).toEqual([]);
  });
});
