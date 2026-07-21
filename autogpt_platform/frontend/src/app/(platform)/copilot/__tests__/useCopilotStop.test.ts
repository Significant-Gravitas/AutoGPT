import { renderHook } from "@testing-library/react";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStop } from "../useCopilotStop";

const mockCancel = vi.fn();
const mockToast = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  postV2CancelSessionTask: (sessionId: string) => mockCancel(sessionId),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (args: unknown) => mockToast(args),
}));

const CANCELLED_MARKER_TEXT = "[__COPILOT_ERROR_f7a1__] Operation cancelled";

type Messages = UIMessage[];

function assistantMessage(text: string): UIMessage {
  return {
    id: `assistant-${text}`,
    role: "assistant",
    parts: [{ type: "text", text, state: "done" }],
  };
}

function userMessage(text: string): UIMessage {
  return {
    id: `user-${text}`,
    role: "user",
    parts: [{ type: "text", text, state: "done" }],
  };
}

/** Build a `stop` handler with capturable refs/spies so each test can assert
 * on the flags and messages the handler mutates. */
function makeHarness({
  sessionId = "session-1",
  prev = [assistantMessage("streaming")] as Messages,
  refetchSession = vi.fn(() => Promise.resolve({ data: undefined })),
  sdkStop = vi.fn(),
}: {
  sessionId?: string | null;
  prev?: Messages;
  refetchSession?: () => Promise<{ data?: unknown }>;
  sdkStop?: () => void;
} = {}) {
  const isUserStoppingRef = { current: false };
  const isCancelInFlightRef = { current: false };
  const setIsUserStopping = vi.fn();
  let captured: Messages = prev;
  const setMessages = vi.fn(
    (updater: Messages | ((p: Messages) => Messages)) => {
      captured = typeof updater === "function" ? updater(captured) : updater;
    },
  );

  const { result } = renderHook(() =>
    useCopilotStop({
      sessionId,
      sdkStop,
      setMessages: setMessages as never,
      isUserStoppingRef,
      setIsUserStopping,
      isCancelInFlightRef,
      refetchSession,
    }),
  );

  return {
    stop: result.current,
    isUserStoppingRef,
    isCancelInFlightRef,
    setIsUserStopping,
    setMessages,
    getMessages: () => captured,
  };
}

function activeStreamResult(active: boolean) {
  return {
    data: { status: 200, data: { active_stream: active } },
  };
}

describe("useCopilotStop", () => {
  beforeEach(() => {
    mockCancel.mockReset();
    mockToast.mockReset();
    mockCancel.mockResolvedValue({ status: 200, data: {} });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("flags the stop, aborts the SDK fetch, and appends the cancellation marker", async () => {
    const sdkStop = vi.fn();
    const h = makeHarness({ sdkStop });

    await h.stop();

    expect(sdkStop).toHaveBeenCalledTimes(1);
    expect(h.setIsUserStopping).toHaveBeenCalledWith(true);
    const last = h.getMessages().at(-1)!;
    expect(last.parts.at(-1)).toEqual({
      type: "text",
      text: CANCELLED_MARKER_TEXT,
    });
  });

  it("swallows sdkStop throwing when no fetch is in flight", async () => {
    const sdkStop = vi.fn(() => {
      throw new Error("no active fetch");
    });
    const h = makeHarness({ sdkStop });

    await expect(h.stop()).resolves.toBeUndefined();
    expect(h.setIsUserStopping).toHaveBeenCalledWith(true);
  });

  it("does not append a marker when the last message is not an assistant", async () => {
    const prev = [assistantMessage("earlier"), userMessage("latest")];
    const h = makeHarness({ prev });

    await h.stop();

    const last = h.getMessages().at(-1)!;
    expect(last.role).toBe("user");
    expect(
      last.parts.some(
        (p) => p.type === "text" && p.text === CANCELLED_MARKER_TEXT,
      ),
    ).toBe(false);
  });

  it("skips the cancel request and refetch when there is no session", async () => {
    const refetchSession = vi.fn(() => Promise.resolve({ data: undefined }));
    const h = makeHarness({ sessionId: null, refetchSession });

    await h.stop();

    expect(mockCancel).not.toHaveBeenCalled();
    expect(refetchSession).not.toHaveBeenCalled();
    expect(h.isCancelInFlightRef.current).toBe(false);
  });

  it("shows a soft toast when the cancel is published but not yet confirmed", async () => {
    mockCancel.mockResolvedValue({
      status: 200,
      data: { reason: "cancel_published_not_confirmed" },
    });
    const h = makeHarness();

    await h.stop();

    expect(mockCancel).toHaveBeenCalledWith("session-1");
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Stop may take a moment" }),
    );
  });

  it("shows a destructive toast when the cancel request fails", async () => {
    mockCancel.mockRejectedValue(new Error("network down"));
    const h = makeHarness();

    await h.stop();

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Could not stop the task",
        variant: "destructive",
      }),
    );
  });

  it("clears the user-stop flag once the refetch confirms no active stream", async () => {
    const refetchSession = vi.fn(() =>
      Promise.resolve(activeStreamResult(false)),
    );
    const h = makeHarness({ refetchSession });

    await h.stop();

    expect(refetchSession).toHaveBeenCalledTimes(1);
    expect(h.isUserStoppingRef.current).toBe(false);
    expect(h.setIsUserStopping).toHaveBeenLastCalledWith(false);
    expect(h.isCancelInFlightRef.current).toBe(false);
  });

  it("keeps the user-stop flag set while the backend still reports an active stream", async () => {
    const refetchSession = vi.fn(() =>
      Promise.resolve(activeStreamResult(true)),
    );
    const h = makeHarness({ refetchSession });

    await h.stop();

    expect(h.isUserStoppingRef.current).toBe(true);
    expect(h.setIsUserStopping).not.toHaveBeenCalledWith(false);
    // The guard is always released so the fallback reset effect can run later.
    expect(h.isCancelInFlightRef.current).toBe(false);
  });

  it("leaves the user-stop flag set but releases the guard when the refetch throws", async () => {
    const refetchSession = vi.fn(() =>
      Promise.reject(new Error("refetch failed")),
    );
    const h = makeHarness({ refetchSession });

    await h.stop();

    expect(h.isUserStoppingRef.current).toBe(true);
    expect(h.isCancelInFlightRef.current).toBe(false);
  });
});
