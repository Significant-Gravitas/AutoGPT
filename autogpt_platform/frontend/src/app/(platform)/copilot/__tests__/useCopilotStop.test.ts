import { renderHook } from "@testing-library/react";
import type { UIMessage } from "ai";
import { useRef } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));

const mockCancel =
  vi.fn<(sessionId: string) => Promise<{ status: number; data: unknown }>>();
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  postV2CancelSessionTask: (sessionId: string) => mockCancel(sessionId),
}));

import { useCopilotStop } from "../useCopilotStop";

function asstMessage(parts: UIMessage["parts"], id = "a1"): UIMessage {
  return { id, role: "assistant", parts };
}

interface SetupOpts {
  sessionId?: string | null;
  initialMessages?: UIMessage[];
}

function setup(opts: SetupOpts = {}) {
  const sdkStop = vi.fn();
  const setIsUserStopping = vi.fn();
  let messages: UIMessage[] = opts.initialMessages ?? [];
  const setMessages = vi.fn((updater: unknown) => {
    if (typeof updater === "function") {
      messages = (updater as (prev: UIMessage[]) => UIMessage[])(messages);
    } else {
      messages = updater as UIMessage[];
    }
  });

  const { result } = renderHook(() => {
    const isUserStoppingRef = useRef(false);
    return {
      stop: useCopilotStop({
        sessionId: "sessionId" in opts ? opts.sessionId! : "sess-1",
        sdkStop,
        setMessages,
        isUserStoppingRef,
        setIsUserStopping,
      }),
      isUserStoppingRef,
    };
  });

  return {
    stop: () => result.current.stop(),
    isUserStoppingRef: () => result.current.isUserStoppingRef,
    sdkStop,
    setMessages,
    setIsUserStopping,
    getMessages: () => messages,
  };
}

describe("useCopilotStop", () => {
  beforeEach(() => {
    mockToast.mockClear();
    mockCancel.mockReset();
  });

  it("flips the user-stop flags synchronously and aborts the SDK stream", async () => {
    mockCancel.mockResolvedValue({ status: 200, data: { reason: "ok" } });
    const { stop, sdkStop, setIsUserStopping, isUserStoppingRef } = setup();

    await stop();

    expect(isUserStoppingRef().current).toBe(true);
    expect(setIsUserStopping).toHaveBeenCalledWith(true);
    expect(sdkStop).toHaveBeenCalledTimes(1);
  });

  it("appends the cancellation marker to the trailing assistant message", async () => {
    mockCancel.mockResolvedValue({ status: 200, data: { reason: "ok" } });
    const { stop, getMessages } = setup({
      initialMessages: [
        asstMessage([{ type: "text", text: "partial reply", state: "done" }]),
      ],
    });

    await stop();
    const after = getMessages();
    const last = after[after.length - 1];
    const tail = last.parts[last.parts.length - 1];
    expect(tail.type).toBe("text");
    expect((tail as { text: string }).text).toContain("Operation cancelled");
  });

  it("leaves messages unchanged when there is no trailing assistant message", async () => {
    mockCancel.mockResolvedValue({ status: 200, data: { reason: "ok" } });
    const { stop, getMessages } = setup({
      initialMessages: [
        {
          id: "u1",
          role: "user",
          parts: [{ type: "text", text: "hi", state: "done" }],
        },
      ],
    });

    await stop();
    const after = getMessages();
    expect(after).toHaveLength(1);
    expect(after[0].role).toBe("user");
  });

  it("swallows sdkStop errors but still proceeds with cancel", async () => {
    mockCancel.mockResolvedValue({ status: 200, data: { reason: "ok" } });
    const { stop, sdkStop } = setup();
    sdkStop.mockImplementation(() => {
      throw new Error("no fetch in flight");
    });

    await expect(stop()).resolves.toBeUndefined();
    expect(mockCancel).toHaveBeenCalledTimes(1);
  });

  it("does not call the cancel endpoint when there is no active sessionId", async () => {
    const { stop } = setup({ sessionId: null });
    await stop();
    expect(mockCancel).not.toHaveBeenCalled();
  });

  it("toasts a 'Stop may take a moment' notice on cancel_published_not_confirmed", async () => {
    mockCancel.mockResolvedValue({
      status: 200,
      data: { reason: "cancel_published_not_confirmed" },
    });
    const { stop } = setup();
    await stop();
    expect(mockToast).toHaveBeenCalledTimes(1);
    expect(mockToast.mock.calls[0][0]).toMatchObject({
      title: "Stop may take a moment",
    });
  });

  it("toasts a destructive notice when the cancel request rejects", async () => {
    mockCancel.mockRejectedValue(new Error("network down"));
    const { stop } = setup();
    await stop();
    expect(mockToast).toHaveBeenCalledTimes(1);
    expect(mockToast.mock.calls[0][0]).toMatchObject({
      title: "Could not stop the task",
      variant: "destructive",
    });
  });

  it("does not toast when the cancel succeeds with a normal reason", async () => {
    mockCancel.mockResolvedValue({ status: 200, data: { reason: "ok" } });
    const { stop } = setup();
    await stop();
    expect(mockToast).not.toHaveBeenCalled();
  });
});
