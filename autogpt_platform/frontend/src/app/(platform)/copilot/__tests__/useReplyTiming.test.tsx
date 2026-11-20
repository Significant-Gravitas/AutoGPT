import { act, cleanup, renderHook } from "@testing-library/react";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  QUIET_TURN_GRACE_MS,
  startReplyTiming,
  type ReplyView,
} from "../helpers/replyTiming";
import { recordReplySpan } from "../helpers/replyTimingSpan";
import { useReplyTiming } from "../useReplyTiming";

vi.mock("../helpers/replyTimingSpan", () => ({ recordReplySpan: vi.fn() }));

const VIEW: ReplyView = {
  sessionId: "session-1",
  messages: [],
  status: "ready",
  isSettled: true,
  isQueued: false,
  isUserStopping: false,
};
const SENT: UIMessage = {
  id: "sent-1",
  role: "user",
  parts: [{ type: "text", text: "hello" }],
};

function renderTiming(sendMessage = vi.fn(async () => {})) {
  return renderHook(
    (view: ReplyView) =>
      useReplyTiming({ ...view, streamClient: "ai-sdk", sendMessage }),
    { initialProps: VIEW },
  );
}

async function advance(ms: number) {
  await act(() => vi.advanceTimersByTimeAsync(ms));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "performance"] });
});

afterEach(async () => {
  cleanup();
  await advance(1);
  vi.clearAllTimers();
  vi.useRealTimers();
});

describe("reply timing lifecycle", () => {
  it("keeps one recheck across renders without extending the grace period", async () => {
    const { result, rerender } = renderTiming();
    await act(() =>
      result.current.sendMessage(
        { text: "hello" },
        startReplyTiming({ isNewChat: false, isExpert: false }),
      ),
    );
    await advance(0);
    expect(vi.getTimerCount()).toBe(1);

    await advance(100);
    rerender({ ...VIEW });
    rerender({ ...VIEW });
    expect(vi.getTimerCount()).toBe(1);

    await advance(QUIET_TURN_GRACE_MS - 101);
    expect(vi.getTimerCount()).toBe(1);
    await advance(1);
    expect(vi.getTimerCount()).toBe(0);
    expect(recordReplySpan).not.toHaveBeenCalled();
  });

  it("clears a pending recheck when the chat unmounts", async () => {
    const { result, unmount } = renderTiming();
    await act(() =>
      result.current.sendMessage(
        { text: "hello" },
        startReplyTiming({ isNewChat: false, isExpert: false }),
      ),
    );
    await advance(0);
    expect(vi.getTimerCount()).toBe(1);

    unmount();
    await advance(1);
    expect(vi.getTimerCount()).toBe(0);
    expect(recordReplySpan).not.toHaveBeenCalled();
  });

  it("clears the grace timer when answer text arrives", async () => {
    const { result, rerender } = renderTiming();
    await act(() =>
      result.current.sendMessage(
        { text: "hello" },
        startReplyTiming({ isNewChat: false, isExpert: true }),
      ),
    );
    rerender({ ...VIEW, messages: [SENT], isQueued: true });
    rerender({ ...VIEW, messages: [SENT] });
    expect(vi.getTimerCount()).toBe(1);

    rerender({
      ...VIEW,
      messages: [
        SENT,
        {
          id: "reply-1",
          role: "assistant",
          parts: [{ type: "text", text: "hi" }],
        },
      ],
    });
    await advance(0);
    expect(vi.getTimerCount()).toBe(0);
    expect(recordReplySpan).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({
        outcome: "answered",
        wasQueued: true,
        start: expect.objectContaining({ isExpert: true }),
      }),
    );
  });

  it("does not schedule a recheck when a send settles after unmount", async () => {
    let resolveSend = () => {};
    const pending = new Promise<void>((resolve) => {
      resolveSend = resolve;
    });
    const { result, rerender, unmount } = renderTiming(vi.fn(() => pending));
    let sent: Promise<void>;
    act(() => {
      sent = result.current.sendMessage(
        { text: "hello" },
        startReplyTiming({ isNewChat: false, isExpert: false }),
      );
    });
    rerender({
      ...VIEW,
      messages: [SENT],
      status: "streaming",
      isSettled: false,
    });
    unmount();
    await advance(1);
    expect(recordReplySpan).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ outcome: "abandoned" }),
    );

    await act(async () => {
      resolveSend();
      await sent;
    });
    expect(vi.getTimerCount()).toBe(0);
  });
});
