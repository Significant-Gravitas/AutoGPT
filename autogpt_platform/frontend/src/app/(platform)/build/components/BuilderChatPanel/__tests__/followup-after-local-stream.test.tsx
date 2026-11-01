import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import type { ChatStatus } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// A follow-up sent while the panel still draws Otto's answer, but after the
// backend's turn has ended: `/messages/pending` refuses it, and the panel
// used to report that to Sentry and hand the composer a failure toast.

const queueFollowUpMessage = vi.hoisted(() => vi.fn());
const captureException = vi.hoisted(() => vi.fn());

vi.mock("@/app/(platform)/copilot/helpers/queueFollowUpMessage", () => ({
  queueFollowUpMessage,
}));
vi.mock("@sentry/nextjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@sentry/nextjs")>()),
  captureException,
}));

import { useBuilderChatSend } from "../useBuilderChatSend";

interface StreamState {
  status: ChatStatus;
  isReconnecting: boolean;
  isFinishProbing: boolean;
}

const sendMessage = vi.fn();
const queueMessage = vi.fn();

function renderSend(initial: Partial<StreamState> = {}) {
  return renderHook(
    (stream: StreamState) =>
      useBuilderChatSend({
        sessionId: "session-1",
        sendMessage,
        queueMessage,
        ...stream,
      }),
    {
      initialProps: {
        status: "streaming",
        isReconnecting: false,
        isFinishProbing: false,
        ...initial,
      },
    },
  );
}

function noActiveTurn() {
  const error = new Error("Session has no active turn to queue against.");
  error.name = "QueueFollowUpNotActiveError";
  return error;
}

async function flush() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

beforeEach(() => {
  queueFollowUpMessage.mockRejectedValue(noActiveTurn());
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("useBuilderChatSend — follow-up refused by the backend", () => {
  it("sends it as a new turn once the panel's stream settles", async () => {
    const { result, rerender } = renderSend();

    let settled = false;
    const send = result.current.onSend("Now add an email step");
    void send.then(() => (settled = true));
    await waitFor(() => expect(queueFollowUpMessage).toHaveBeenCalledTimes(1));

    expect(queueFollowUpMessage).toHaveBeenCalledWith(
      "session-1",
      "Now add an email step",
    );
    expect(sendMessage).not.toHaveBeenCalled();
    expect(settled).toBe(false);

    rerender({
      status: "ready",
      isReconnecting: false,
      isFinishProbing: false,
    });
    await send;

    expect(sendMessage).toHaveBeenCalledTimes(1);
    expect(sendMessage).toHaveBeenCalledWith({ text: "Now add an email step" });
    expect(queueMessage).not.toHaveBeenCalled();
    expect(captureException).not.toHaveBeenCalled();
  });

  it("waits out the post-finish probe and a reconnect as well", async () => {
    const { result, rerender } = renderSend();
    const send = result.current.onSend("and then?");
    await flush();

    rerender({ status: "ready", isReconnecting: false, isFinishProbing: true });
    await flush();
    expect(sendMessage).not.toHaveBeenCalled();

    rerender({ status: "ready", isReconnecting: true, isFinishProbing: false });
    await flush();
    expect(sendMessage).not.toHaveBeenCalled();

    rerender({
      status: "ready",
      isReconnecting: false,
      isFinishProbing: false,
    });
    await send;
    expect(sendMessage).toHaveBeenCalledTimes(1);
  });

  it("queues a second refused follow-up behind the first instead of starting two turns", async () => {
    const { result, rerender } = renderSend();
    const first = result.current.onSend("first");
    const second = result.current.onSend("second");
    await waitFor(() => expect(queueFollowUpMessage).toHaveBeenCalledTimes(2));

    // Once the first has gone out, the backend has a turn to queue against.
    queueFollowUpMessage.mockResolvedValue({
      buffer_length: 1,
      max_buffer_length: 10,
      turn_in_flight: true,
    });
    rerender({
      status: "ready",
      isReconnecting: false,
      isFinishProbing: false,
    });
    await Promise.all([first, second]);

    expect(sendMessage).toHaveBeenCalledTimes(1);
    expect(sendMessage).toHaveBeenCalledWith({ text: "first" });
    expect(queueMessage).toHaveBeenCalledWith("second");
  });

  it("gives up the send when the panel unmounts mid-wait", async () => {
    const { result, unmount } = renderSend();
    const send = result.current.onSend("and then?");
    await flush();

    unmount();
    await expect(send).resolves.toBeUndefined();
    expect(sendMessage).not.toHaveBeenCalled();
  });
});

describe("useBuilderChatSend — other paths", () => {
  it("queues the follow-up when the backend accepts it", async () => {
    queueFollowUpMessage.mockResolvedValue({
      buffer_length: 1,
      max_buffer_length: 10,
      turn_in_flight: true,
    });
    const { result } = renderSend();
    await result.current.onSend("and then?");

    expect(queueMessage).toHaveBeenCalledWith("and then?");
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("reports any other queue failure and lets the composer show it", async () => {
    const failure = new Error("Expected 200 from /messages/pending; got 500");
    queueFollowUpMessage.mockRejectedValue(failure);
    const { result } = renderSend();

    await expect(result.current.onSend("and then?")).rejects.toBe(failure);
    expect(captureException).toHaveBeenCalledWith(failure);
    expect(queueMessage).not.toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("sends directly when no turn is in flight", async () => {
    const { result } = renderSend({ status: "ready" });
    await result.current.onSend("  Build me a workflow  ");

    expect(queueFollowUpMessage).not.toHaveBeenCalled();
    expect(sendMessage).toHaveBeenCalledWith({ text: "Build me a workflow" });
  });
});
