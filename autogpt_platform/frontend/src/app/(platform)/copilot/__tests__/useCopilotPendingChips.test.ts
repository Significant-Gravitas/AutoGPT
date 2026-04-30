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

  it("queueMessage adds a chip locally without hitting backend", () => {
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
      result.current.queueMessage("hello");
    });
    expect(result.current.queuedMessages).toEqual(["hello"]);
  });

  it("promotes one bubble per chip when a second new assistant appears (auto-continue)", () => {
    const setMessages = vi.fn();

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

    // User queues two distinct chips while turn 1 is still streaming.
    act(() => {
      result.current.queueMessage("followup-a");
      result.current.queueMessage("followup-b");
    });

    setMessages.mockClear();

    // A SECOND new assistant id appears → auto-continue detected.
    rerender({
      messages: [
        user("u1"),
        assistant("a1", "turn1"),
        assistant("a2", "turn2"),
      ],
    });

    const promoteCall = setMessages.mock.calls.find(
      ([fn]) => typeof fn === "function",
    );
    expect(promoteCall).toBeDefined();

    const updater = promoteCall![0] as (prev: UIMessage[]) => UIMessage[];
    const after = updater([user("u1"), assistant("a1"), assistant("a2")]);

    // One bubble per chip — cardinality preserved.
    const promoted = after.filter((m) =>
      m.id.startsWith("promoted-auto-continue-"),
    );
    expect(promoted).toHaveLength(2);
    expect(
      promoted.map((m) => (m.parts![0] as { text?: string }).text),
    ).toEqual(["followup-a", "followup-b"]);

    // Both bubbles inserted before a2.
    const a2Idx = after.findIndex((m) => m.id === "a2");
    const promotedIndices = promoted.map((m) =>
      after.findIndex((x) => x.id === m.id),
    );
    expect(promotedIndices.every((i) => i < a2Idx)).toBe(true);

    // Chips cleared.
    expect(result.current.queuedMessages).toEqual([]);
  });

  it("turn-start drain: peek count=0 clears chips once submitted→streaming", async () => {
    const setMessages = vi.fn();
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
      result.current.queueMessage("survives");
    });

    peekMock.mockRejectedValue(new Error("network blip"));

    await act(async () => {
      vi.advanceTimersByTime(2_000);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(result.current.queuedMessages).toEqual(["survives"]);
    const promoteCall = setMessages.mock.calls.find(([arg]) => {
      if (typeof arg !== "function") return false;
      const updated = (arg as (p: UIMessage[]) => UIMessage[])([]);
      return updated.some((m) => m.id.startsWith("promoted-"));
    });
    expect(promoteCall).toBeUndefined();
  });

  it("mid-turn poll promotes one bubble per drained chip", async () => {
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
      result.current.queueMessage("chipA");
      result.current.queueMessage("chipB");
    });

    peekMock.mockResolvedValue({
      status: 200,
      data: { count: 0, messages: [] },
    });

    await act(async () => {
      vi.advanceTimersByTime(2_000);
      await Promise.resolve();
      await Promise.resolve();
    });

    const promotedCall = setMessages.mock.calls.find(([arg]) => {
      if (typeof arg !== "function") return false;
      const updated = (arg as (p: UIMessage[]) => UIMessage[])([]);
      return updated.some((m) => m.id.startsWith("promoted-midturn-"));
    });
    expect(promotedCall).toBeDefined();

    const streamingUpdater = promotedCall![0] as (
      prev: UIMessage[],
    ) => UIMessage[];
    const priorMessages = [user("u1"), assistant("a1", "streaming...")];
    const afterPromotion = streamingUpdater(priorMessages);

    // Two chips → two bubbles, both inserted before the streaming assistant.
    expect(afterPromotion).toHaveLength(4);
    const lastIdx = afterPromotion.length - 1;
    expect(afterPromotion[lastIdx].role).toBe("assistant");
    expect(afterPromotion[lastIdx].id).toBe("a1");
    const promoted = afterPromotion.filter((m) =>
      m.id.startsWith("promoted-midturn-"),
    );
    expect(promoted).toHaveLength(2);
    expect(
      promoted.map((m) => (m.parts![0] as { text?: string }).text),
    ).toEqual(["chipA", "chipB"]);

    expect(result.current.queuedMessages).toEqual([]);
  });

  it("mid-turn poll: a peek that resolves after the user switches sessions does not promote into the new session", async () => {
    vi.useFakeTimers();
    const setMessagesA = vi.fn();
    const setMessagesB = vi.fn();
    type Props = { sessionId: string; setMessages: typeof setMessagesA };
    const { result, rerender } = renderHook<
      ReturnType<typeof useCopilotPendingChips>,
      Props
    >(
      ({ sessionId, setMessages }) =>
        useCopilotPendingChips({
          sessionId,
          status: "streaming",
          messages: [user("u1"), assistant("a1")],
          setMessages,
        }),
      {
        initialProps: { sessionId: "sess-A", setMessages: setMessagesA },
      },
    );

    act(() => {
      result.current.queueMessage("chipA");
    });

    // Poll fires for sess-A but resolves AFTER we've switched to sess-B.
    let resolvePeek!: (value: unknown) => void;
    peekMock.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolvePeek = resolve;
        }),
    );

    await act(async () => {
      vi.advanceTimersByTime(2_000);
    });

    // User switches sessions.
    rerender({ sessionId: "sess-B", setMessages: setMessagesB });

    // Old poll resolves now — backend says count=0 — but we've already
    // moved on to sess-B.  The session-id guard must short-circuit so
    // setMessagesA is never called for the promotion path.
    setMessagesA.mockClear();
    await act(async () => {
      resolvePeek({ status: 200, data: { count: 0, messages: [] } });
      await Promise.resolve();
      await Promise.resolve();
    });

    const promotedCallA = setMessagesA.mock.calls.find(([arg]) => {
      if (typeof arg !== "function") return false;
      const updated = (arg as (p: UIMessage[]) => UIMessage[])([]);
      return updated.some((m) => m.id.startsWith("promoted-"));
    });
    expect(promotedCallA).toBeUndefined();
  });

  it("mid-turn poll: chip appended during in-flight poll survives the drain", async () => {
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
      result.current.queueMessage("chipA");
    });

    // Backend has drained chipA (count=0). The poll's GET resolves, but
    // the user appends chipB before the setState commits.
    let resolvePeek!: (value: unknown) => void;
    peekMock.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolvePeek = resolve;
        }),
    );

    await act(async () => {
      vi.advanceTimersByTime(2_000);
    });

    // While the GET is still pending, append a second chip.
    act(() => {
      result.current.queueMessage("chipB");
    });

    // Now resolve the in-flight peek.
    await act(async () => {
      resolvePeek({ status: 200, data: { count: 0, messages: [] } });
      await Promise.resolve();
      await Promise.resolve();
    });

    // chipA was drained → promoted; chipB survived as a chip.
    expect(result.current.queuedMessages).toEqual(["chipB"]);
  });
});
