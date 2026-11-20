import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { getV2GetPendingMessages } from "@/app/api/__generated__/endpoints/chat/chat";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import type { ChatStatus, UIMessage } from "ai";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useBuilderChatSend } from "../useBuilderChatSend";

// A follow-up sent while the panel still draws Otto's answer, after the
// backend's turn has ended, is refused by `/messages/pending`. The panel must
// show it as a queued chip, send it as a new turn once its own stream has
// settled, and never lose it: opening another workflow hands it back to the
// composer at once, and one the builder unmounted on comes back there the
// next time its chat opens.

const queueFollowUpMessage = vi.hoisted(() => vi.fn());
const captureException = vi.hoisted(() => vi.fn());
const toast = vi.hoisted(() => vi.fn());

vi.mock("@/app/(platform)/copilot/helpers/queueFollowUpMessage", () => ({
  queueFollowUpMessage,
}));
vi.mock("@sentry/nextjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@sentry/nextjs")>()),
  captureException,
}));
vi.mock(
  "@/app/api/__generated__/endpoints/chat/chat",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("@/app/api/__generated__/endpoints/chat/chat")
    >()),
    getV2GetPendingMessages: vi.fn(),
  }),
);
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast,
}));

const getPendingMessages = vi.mocked(getV2GetPendingMessages);

const HELD_KEY = "copilot-held-follow-ups:session-1";
const ACCEPTED = {
  buffer_length: 1,
  max_buffer_length: 10,
  turn_in_flight: true,
};

interface StreamState {
  sessionId: string | null;
  status: ChatStatus;
  isReconnecting: boolean;
  isFinishProbing: boolean;
}

const sendMessage = vi.fn();

function useBuilderChat(stream: StreamState) {
  const [messages, setMessages] = useState<UIMessage[]>([]);
  return {
    ...useBuilderChatSend({ ...stream, messages, setMessages, sendMessage }),
    messages,
  };
}

function renderBuilderChat(initial: Partial<StreamState> = {}) {
  let stream: StreamState = {
    sessionId: "session-1",
    status: "streaming",
    isReconnecting: false,
    isFinishProbing: false,
    ...initial,
  };
  const view = renderHook((props: StreamState) => useBuilderChat(props), {
    initialProps: stream,
  });
  return {
    ...view,
    /** Re-renders with the panel's stream or session moved on. */
    update(next: Partial<StreamState>) {
      stream = { ...stream, ...next };
      view.rerender(stream);
    },
    /** Sends the way the composer does; settles when `onSend` does. */
    send(text: string) {
      let sent!: Promise<void>;
      act(() => {
        sent = view.result.current.onSend(text);
      });
      return sent;
    },
  };
}

function pending(messages: string[]) {
  return {
    status: 200,
    data: { count: messages.length, messages },
    headers: new Headers(),
  } as Awaited<ReturnType<typeof getV2GetPendingMessages>>;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, resolve, reject };
}

function noActiveTurn() {
  const error = new Error("Session has no active turn to queue against.");
  error.name = "QueueFollowUpNotActiveError";
  return error;
}

function recoveredDraft() {
  return useCopilotUIStore.getState().recoveredDraft;
}

async function flush() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

beforeEach(() => {
  queueFollowUpMessage.mockRejectedValue(noActiveTurn());
  getPendingMessages.mockResolvedValue(pending([]));
  useCopilotUIStore.getState().clearRecoveredDraft();
});

afterEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
});

describe("useBuilderChatSend — follow-up refused by the backend", () => {
  it("sends it as a new turn once the panel's stream settles", async () => {
    const chat = renderBuilderChat();

    chat.send("Now add an email step");
    await flush();

    expect(queueFollowUpMessage).toHaveBeenCalledExactlyOnceWith(
      "session-1",
      "Now add an email step",
    );
    expect(sendMessage).not.toHaveBeenCalled();

    chat.update({ status: "ready" });
    await waitFor(() =>
      expect(sendMessage).toHaveBeenCalledExactlyOnceWith({
        text: "Now add an email step",
      }),
    );
    expect(captureException).not.toHaveBeenCalled();
  });

  it("shows it as a queued chip while it waits, and frees the composer for the next message", async () => {
    const chat = renderBuilderChat();

    let isResolved = false;
    void chat.send("and then?").then(() => (isResolved = true));
    await flush();

    // The composer keeps Enter and the queue button locked until this
    // resolves, so it must not wait for the stream to settle.
    expect(isResolved).toBe(true);
    expect(sendMessage).not.toHaveBeenCalled();
    expect(chat.result.current.queuedMessages).toEqual(["and then?"]);

    chat.update({ status: "ready" });
    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(chat.result.current.queuedMessages).toEqual([]));
  });

  it("waits out the post-finish probe and a reconnect as well", async () => {
    const chat = renderBuilderChat();
    chat.send("and then?");
    await flush();

    chat.update({ status: "ready", isFinishProbing: true });
    await flush();
    expect(sendMessage).not.toHaveBeenCalled();

    chat.update({ isFinishProbing: false, isReconnecting: true });
    await flush();
    expect(sendMessage).not.toHaveBeenCalled();

    chat.update({ isReconnecting: false });
    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));
  });

  it("queues a second refused follow-up behind the first instead of starting two turns", async () => {
    const chat = renderBuilderChat();
    chat.send("first");
    chat.send("second");
    await flush();
    expect(chat.result.current.queuedMessages).toEqual(["first", "second"]);

    // Once the first has gone out, the backend has a turn to queue against.
    queueFollowUpMessage.mockResolvedValue(ACCEPTED);
    chat.update({ status: "ready" });
    await waitFor(() => expect(queueFollowUpMessage).toHaveBeenCalledTimes(3));

    expect(sendMessage).toHaveBeenCalledExactlyOnceWith({ text: "first" });
    expect(queueFollowUpMessage).toHaveBeenLastCalledWith(
      "session-1",
      "second",
    );
    await waitFor(() =>
      expect(chat.result.current.queuedMessages).toEqual(["second"]),
    );
    expect(window.sessionStorage.getItem(HELD_KEY)).toBeNull();
  });

  it("hands a held follow-up back with the error when queueing it behind a new turn fails", async () => {
    const chat = renderBuilderChat();
    chat.send("first");
    chat.send("second");
    await flush();

    const failure = new Error("Expected 200 from /messages/pending; got 500");
    queueFollowUpMessage.mockRejectedValue(failure);
    chat.update({ status: "ready" });
    await waitFor(() => expect(recoveredDraft()).toBe("second"));

    expect(sendMessage).toHaveBeenCalledExactlyOnceWith({ text: "first" });
    expect(captureException).toHaveBeenCalledExactlyOnceWith(failure);
    expect(toast).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ title: "Couldn't send message" }),
    );
    await waitFor(() => expect(chat.result.current.queuedMessages).toEqual([]));
    expect(window.sessionStorage.getItem(HELD_KEY)).toBeNull();
  });

  it("puts it back in the composer the next time its chat opens when the builder unmounts first", async () => {
    const before = renderBuilderChat();
    before.send("and then?");
    await flush();

    before.unmount();
    await flush();
    expect(sendMessage).not.toHaveBeenCalled();
    expect(recoveredDraft()).toBeNull();

    // Coming back to the workflow, or reloading it, opens its chat again.
    renderBuilderChat({ status: "ready" });
    await waitFor(() => expect(recoveredDraft()).toBe("and then?"));
    expect(toast).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ title: "Follow-up not sent" }),
    );
    expect(window.sessionStorage.getItem(HELD_KEY)).toBeNull();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("keeps it for its own workflow's chat when the panel opens another workflow", async () => {
    const chat = renderBuilderChat();
    chat.send("and then?");
    await flush();

    // The next workflow's chat is idle, so its stream reads as settled in
    // the very commit that switches to it.
    chat.update({ sessionId: "session-2", status: "ready" });
    await waitFor(() =>
      expect(toast).toHaveBeenCalledExactlyOnceWith(
        expect.objectContaining({ title: "Follow-up not sent" }),
      ),
    );

    // Not in this workflow's composer, where Enter would send it here.
    expect(recoveredDraft()).toBeNull();
    expect(sendMessage).not.toHaveBeenCalled();
    expect(chat.result.current.queuedMessages).toEqual([]);
    expect(window.sessionStorage.getItem(HELD_KEY)).toContain("and then?");

    chat.update({ sessionId: "session-1" });
    await waitFor(() => expect(recoveredDraft()).toBe("and then?"));
    expect(window.sessionStorage.getItem(HELD_KEY)).toBeNull();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("gives it back on its own workflow when the panel leaves before any stream settles", async () => {
    const chat = renderBuilderChat();
    chat.send("and then?");
    await flush();

    chat.update({ sessionId: "session-2" });
    await waitFor(() => expect(toast).toHaveBeenCalledTimes(1));
    expect(recoveredDraft()).toBeNull();

    // Back on the first workflow, its stream settles with nothing to send.
    chat.update({ sessionId: "session-1", status: "ready" });
    await waitFor(() => expect(recoveredDraft()).toBe("and then?"));
    await flush();
    expect(sendMessage).not.toHaveBeenCalled();
    expect(toast).toHaveBeenCalledTimes(2);
  });

  it("keeps a held follow-up the backend refuses again, and sends it after the next turn", async () => {
    const chat = renderBuilderChat();
    chat.send("first");
    chat.send("second");
    await flush();

    // "first" goes out as a new turn; "second" re-queues behind it, and the
    // backend refuses it again before that turn registers.
    chat.update({ status: "ready" });
    await waitFor(() => expect(queueFollowUpMessage).toHaveBeenCalledTimes(3));
    expect(sendMessage).toHaveBeenCalledExactlyOnceWith({ text: "first" });
    await waitFor(() =>
      expect(chat.result.current.queuedMessages).toEqual(["second"]),
    );
    expect(window.sessionStorage.getItem(HELD_KEY)).toContain("second");

    chat.update({ status: "streaming" });
    chat.update({ status: "ready" });
    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(2));
    expect(sendMessage).toHaveBeenLastCalledWith({ text: "second" });
    await waitFor(() => expect(chat.result.current.queuedMessages).toEqual([]));
    expect(window.sessionStorage.getItem(HELD_KEY)).toBeNull();
  });

  it("keeps a held follow-up for its own chat when its queue request fails after the panel moved on", async () => {
    const chat = renderBuilderChat();
    chat.send("first");
    chat.send("second");
    await flush();

    const request = deferred<typeof ACCEPTED>();
    queueFollowUpMessage.mockReturnValue(request.promise);
    chat.update({ status: "ready" });
    await waitFor(() => expect(queueFollowUpMessage).toHaveBeenCalledTimes(3));

    chat.update({ sessionId: "session-2" });
    const failure = new Error("Expected 200 from /messages/pending; got 500");
    await act(async () => request.reject(failure));
    await flush();

    expect(captureException).toHaveBeenCalledExactlyOnceWith(failure);
    expect(toast).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ title: "Couldn't send message" }),
    );
    expect(recoveredDraft()).toBeNull();
    expect(window.sessionStorage.getItem(HELD_KEY)).toContain("second");
  });
});

describe("useBuilderChatSend — queued chips", () => {
  it("shows a follow-up the backend took once, when a peek reads it before the request answers", async () => {
    const request = deferred<typeof ACCEPTED>();
    queueFollowUpMessage.mockReturnValue(request.promise);
    const peek = deferred<ReturnType<typeof pending>>();
    getPendingMessages.mockReturnValue(peek.promise);
    const chat = renderBuilderChat({ status: "submitted" });

    chat.send("and then?");
    chat.update({ status: "streaming" });
    expect(getPendingMessages).toHaveBeenCalledTimes(1);
    await act(async () => peek.resolve(pending(["and then?"])));
    await act(async () => request.resolve(ACCEPTED));

    expect(chat.result.current.queuedMessages).toEqual(["and then?"]);
  });

  it("takes back the chip, and the bubble a drain promoted it to, when the backend refuses it", async () => {
    const request = deferred<never>();
    queueFollowUpMessage.mockReturnValue(request.promise);
    const chat = renderBuilderChat({ status: "submitted" });

    chat.send("and then?");
    // The turn-start peek finds the buffer empty and takes the chip for
    // drained, before the backend has answered its request.
    chat.update({ status: "streaming" });
    await waitFor(() => expect(chat.result.current.messages).toHaveLength(1));
    await act(async () => request.reject(noActiveTurn()));

    expect(chat.result.current.messages).toEqual([]);
    expect(chat.result.current.queuedMessages).toEqual(["and then?"]);
  });
});

describe("useBuilderChatSend — other paths", () => {
  it("queues the follow-up when the backend accepts it", async () => {
    queueFollowUpMessage.mockResolvedValue(ACCEPTED);
    const chat = renderBuilderChat();

    await act(() => chat.result.current.onSend("and then?"));

    expect(chat.result.current.queuedMessages).toEqual(["and then?"]);
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("reports any other queue failure and lets the composer show it", async () => {
    const failure = new Error("Expected 200 from /messages/pending; got 500");
    queueFollowUpMessage.mockRejectedValue(failure);
    const chat = renderBuilderChat();

    await act(async () => {
      await expect(chat.result.current.onSend("and then?")).rejects.toBe(
        failure,
      );
    });
    expect(captureException).toHaveBeenCalledWith(failure);
    expect(chat.result.current.queuedMessages).toEqual([]);
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("sends directly when no turn is in flight", async () => {
    const chat = renderBuilderChat({ status: "ready" });

    await act(() => chat.result.current.onSend("  Build me a workflow  "));

    expect(queueFollowUpMessage).not.toHaveBeenCalled();
    expect(sendMessage).toHaveBeenCalledWith({ text: "Build me a workflow" });
  });
});
