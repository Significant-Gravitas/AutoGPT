import { act, cleanup, renderHook } from "@testing-library/react";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// --- Toast mock (must be stable across rerenders) ---
const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));

// --- Environment mock ---
vi.mock("@/services/environment", () => ({
  environment: { getAGPTServerBaseUrl: () => "http://localhost:8006" },
}));

// --- API endpoints mock ---
const mockCancelSessionTask = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  postV2CancelSessionTask: (...args: unknown[]) =>
    mockCancelSessionTask(...args),
  getGetV2GetCopilotUsageQueryKey: () => ["usage"],
  getGetV2GetSessionQueryKey: (id: string) => ["session", id],
}));

// --- React Query mock ---
const mockInvalidateQueries = vi.fn();
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: mockInvalidateQueries }),
}));

// --- Helpers mock ---
const mockHasActiveBackendStream = vi.fn(
  (_result: { data?: unknown }) => false,
);
const mockDisconnectSessionStream = vi.fn((_sid: string) => {});
vi.mock("../helpers", () => ({
  getCopilotAuthHeaders: vi.fn(async () => ({ Authorization: "Bearer test" })),
  deduplicateMessages: (msgs: UIMessage[]) => msgs,
  extractSendMessageText: (arg: unknown) =>
    arg && typeof arg === "object" && "text" in arg
      ? String((arg as { text: string }).text)
      : String(arg ?? ""),
  hasActiveBackendStream: (result: { data?: unknown }) =>
    mockHasActiveBackendStream(result),
  hasVisibleAssistantContent: (messages: UIMessage[]) => {
    const last = messages[messages.length - 1];
    if (last?.role !== "assistant") return false;
    return last.parts.some((part: UIMessage["parts"][number]) => {
      if (part.type === "text" && part.text.trim().length > 0) return true;
      if (part.type === "reasoning" && part.text.trim().length > 0) return true;
      if (part.type.startsWith("tool-")) return true;
      return false;
    });
  },
  hasInProgressAssistantParts: (message: UIMessage | undefined) => {
    if (message?.role !== "assistant") return false;
    return message.parts.some((part: UIMessage["parts"][number]) => {
      if (!("state" in part) || typeof part.state !== "string") return false;
      return (
        part.state === "streaming" ||
        part.state === "input-streaming" ||
        part.state === "input-available"
      );
    });
  },
  resolveInProgressTools: (msgs: UIMessage[]) => msgs,
  resolveInterruptedMessage: (msgs: UIMessage[]) => msgs,
  getSendSuppressionReason: () => null,
  disconnectSessionStream: (sid: string) => mockDisconnectSessionStream(sid),
}));

// --- ai SDK mock (DefaultChatTransport must be constructible) ---
vi.mock("ai", () => ({
  DefaultChatTransport: vi.fn().mockImplementation(function () {
    return {};
  }),
}));

// --- @ai-sdk/react useChat mock with callback capture ---
type OnFinishArgs = { isDisconnect?: boolean; isAbort?: boolean };
type UseChatOptions = {
  id?: string;
  onFinish?: (args: OnFinishArgs) => void | Promise<void>;
  onError?: (e: Error) => void;
};
let capturedUseChatOptions: UseChatOptions | null = null;
const capturedUseChatIds: string[] = [];
const mockResumeStream = vi.fn();
const mockSdkStop = vi.fn();
const mockSdkSendMessage = vi.fn();
const mockSetMessages = vi.fn();
let mockMessages: UIMessage[] = [];
let mockStatus: "ready" | "streaming" | "submitted" | "error" = "ready";

vi.mock("@ai-sdk/react", () => ({
  useChat: (opts: UseChatOptions) => {
    capturedUseChatOptions = opts;
    capturedUseChatIds.push(opts.id ?? "");
    return {
      messages: mockMessages,
      sendMessage: mockSdkSendMessage,
      stop: mockSdkStop,
      status: mockStatus,
      error: undefined,
      setMessages: mockSetMessages,
      resumeStream: mockResumeStream,
    };
  },
}));

// Import after mocks
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useCopilotStream } from "../useCopilotStream";

type Args = Parameters<typeof useCopilotStream>[0];

function makeArgs(overrides: Partial<Args> = {}): Args {
  return {
    sessionId: "sess-1",
    hydratedMessages: undefined,
    hasActiveStream: false,
    refetchSession: vi.fn(async () => ({ data: undefined })),
    copilotMode: undefined,
    copilotModel: undefined,
    ...overrides,
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  mockMessages = [];
  mockStatus = "ready";
  capturedUseChatOptions = null;
  capturedUseChatIds.length = 0;
  mockResumeStream.mockClear();
  mockSdkStop.mockClear();
  mockSdkSendMessage.mockClear();
  mockSetMessages.mockClear();
  mockToast.mockClear();
  mockHasActiveBackendStream.mockReturnValue(false);
  mockDisconnectSessionStream.mockClear();
  mockInvalidateQueries.mockClear();
  // Zustand stores are module singletons — wipe per-session coord state so
  // tests don't leak into each other.
  useCopilotStreamStore.getState().resetAll();
});

afterEach(() => {
  vi.useRealTimers();
  cleanup();
});

describe("useCopilotStream — hydration/resume race (SECRT-2242)", () => {
  it("uses a fresh AI SDK chat instance id on each mount of the same session", () => {
    const first = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-1" }),
    });
    first.unmount();

    renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-1" }),
    });

    expect(capturedUseChatIds).toHaveLength(2);
    expect(capturedUseChatIds[0]).toMatch(/^sess-1:/);
    expect(capturedUseChatIds[1]).toMatch(/^sess-1:/);
    expect(capturedUseChatIds[0]).not.toBe(capturedUseChatIds[1]);
  });

  it("defers resume until hydration completes when hydratedMessages arrives late", () => {
    // Arrange: active backend stream, hydration NOT yet complete.
    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: undefined,
      }),
    });

    // Resume effect runs but must wait — no resumeStream fires yet.
    expect(mockResumeStream).not.toHaveBeenCalled();

    // Hydration completes: hydratedMessages becomes defined.
    rerender(
      makeArgs({
        hasActiveStream: true,
        hydratedMessages: [],
      }),
    );

    // pendingResumeRef should have been flushed exactly once.
    expect(mockResumeStream).toHaveBeenCalledTimes(1);
  });

  it("reports active-session restore until visible content arrives", () => {
    const { result, rerender } = renderHook(
      (args: Args) => useCopilotStream(args),
      {
        initialProps: makeArgs({
          hasActiveStream: true,
          hydratedMessages: [],
        }),
      },
    );

    expect(result.current.isRestoringActiveSession).toBe(true);

    // Status flipping to "streaming" is NOT enough: the latch waits for
    // actual content so the "Retrieving latest messages" spinner stays up
    // through the pre-content window of a GET-resume.
    mockStatus = "streaming";
    rerender(
      makeArgs({
        hasActiveStream: true,
        hydratedMessages: [],
      }),
    );
    expect(result.current.isRestoringActiveSession).toBe(true);

    // Once the stream produces real content, the restore indicator flips off.
    mockMessages = [
      {
        id: "a1",
        role: "assistant",
        parts: [{ type: "text", text: "Hello" }],
      } as unknown as UIMessage,
    ];
    rerender(
      makeArgs({
        hasActiveStream: true,
        hydratedMessages: [],
      }),
    );
    expect(result.current.isRestoringActiveSession).toBe(false);
  });

  it("does not double-resume when hydration arrives after an already-queued resume", () => {
    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: undefined,
      }),
    });

    // Hydration completes once — pending resume flushes.
    rerender(makeArgs({ hasActiveStream: true, hydratedMessages: [] }));
    expect(mockResumeStream).toHaveBeenCalledTimes(1);

    // Subsequent rerender with the same hydration must not re-trigger resume
    // because hasResumedRef is now set for this mount.
    rerender(makeArgs({ hasActiveStream: true, hydratedMessages: [] }));
    expect(mockResumeStream).toHaveBeenCalledTimes(1);
  });

  it("resumes immediately when hydration completed before the active-stream flag flipped on", () => {
    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: false,
        hydratedMessages: [],
      }),
    });

    expect(mockResumeStream).not.toHaveBeenCalled();

    rerender(makeArgs({ hasActiveStream: true, hydratedMessages: [] }));

    expect(mockResumeStream).toHaveBeenCalledTimes(1);
  });

  it("drops a trailing assistant snapshot before full replay resume", () => {
    const userMessage: UIMessage = {
      id: "u1",
      role: "user",
      parts: [{ type: "text", text: "hello", state: "done" }],
    };
    const assistantMessage: UIMessage = {
      id: "a1",
      role: "assistant",
      parts: [{ type: "text", text: "partial", state: "streaming" }],
    };
    mockMessages = [userMessage, assistantMessage];

    renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: [userMessage, assistantMessage],
      }),
    });

    const messageUpdaters = mockSetMessages.mock.calls
      .map(([arg]) => arg)
      .filter((arg) => typeof arg === "function") as Array<
      (messages: UIMessage[]) => UIMessage[]
    >;
    const resumeUpdater = messageUpdaters[messageUpdaters.length - 1];

    expect(resumeUpdater?.([userMessage, assistantMessage])).toEqual([
      userMessage,
    ]);
    expect(mockResumeStream).toHaveBeenCalledTimes(1);
  });

  it("keeps a completed trailing assistant before full replay resume", () => {
    const userMessage: UIMessage = {
      id: "u1",
      role: "user",
      parts: [{ type: "text", text: "hello", state: "done" }],
    };
    const assistantMessage: UIMessage = {
      id: "a1",
      role: "assistant",
      parts: [{ type: "text", text: "final", state: "done" }],
    };
    mockMessages = [userMessage, assistantMessage];

    renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: [userMessage, assistantMessage],
      }),
    });

    const messageUpdaters = mockSetMessages.mock.calls
      .map(([arg]) => arg)
      .filter((arg) => typeof arg === "function") as Array<
      (messages: UIMessage[]) => UIMessage[]
    >;
    const resumeUpdater = messageUpdaters[messageUpdaters.length - 1];

    expect(resumeUpdater?.([userMessage, assistantMessage])).toEqual([
      userMessage,
      assistantMessage,
    ]);
    expect(mockResumeStream).toHaveBeenCalledTimes(1);
  });
});

describe("useCopilotStream — unmount cleanup", () => {
  it("stops the in-flight fetch and disconnects backend listeners on unmount", () => {
    const { unmount } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-A" }),
    });

    // Start a reconnect cycle so there's state to tear down.
    // (handleReconnect arms a timer; unmount should not resume the stream.)
    void capturedUseChatOptions!.onFinish!({ isDisconnect: true });

    unmount();

    // Both cleanup actions must fire: abort the fetch + tell the backend
    // to release its XREAD listeners without waiting for the timeout.
    expect(mockSdkStop).toHaveBeenCalled();
    expect(mockDisconnectSessionStream).toHaveBeenCalledWith("sess-A");
  });

  it("reconnect timer does not fire resumeStream after unmount", async () => {
    const { unmount } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-A" }),
    });

    await act(async () => {
      await capturedUseChatOptions!.onFinish!({ isDisconnect: true });
    });

    // Parent would remount us with a new session key — simulate the
    // unmount that happens before the reconnect timer backoff expires.
    unmount();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });

    expect(mockResumeStream).not.toHaveBeenCalled();
  });
});

describe("useCopilotStream — forced reconnect timeout (SECRT-2241)", () => {
  it("forces UI back to idle after 30s of continuous reconnection", async () => {
    const refetchSession = vi.fn(async () => ({ data: undefined }));
    const { result } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ refetchSession }),
    });

    // Kick off reconnect via a disconnect finish.
    await act(async () => {
      await capturedUseChatOptions!.onFinish!({ isDisconnect: true });
    });

    // Initially reconnect is scheduled — but we won't let it succeed. Instead
    // advance exactly 30s so the forced-timeout callback fires.
    mockToast.mockClear();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    // Forced timeout should have fired its toast.
    const timeoutToast = mockToast.mock.calls.find(
      ([call]) => (call as { title?: string }).title === "Connection timed out",
    );
    expect(timeoutToast).toBeDefined();

    // isReconnecting should flip back to false once reconnectExhausted is set.
    expect(result.current.isReconnecting).toBe(false);
  });

  it("forced-timeout does not fire after unmount", async () => {
    const { unmount } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs(),
    });

    await act(async () => {
      await capturedUseChatOptions!.onFinish!({ isDisconnect: true });
    });

    // Parent remounts on session switch — the old mount unmounts. Any
    // pending timers belong to that mount and must not fire toasts after
    // its teardown.
    unmount();

    mockToast.mockClear();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    const timeoutToast = mockToast.mock.calls.find(
      ([call]) => (call as { title?: string }).title === "Connection timed out",
    );
    expect(timeoutToast).toBeUndefined();
  });
});

// Cursor tracking for incremental resume was removed — resume always replays
// from "0-0" because AI SDK v5's UIMessageStream parser throws on orphan
// `*-delta` / `*-end` chunks, and a cursor-based XREAD skips the envelope +
// `*-start` chunks that precede the cursor. See copilotStreamTransport.ts.

describe("useCopilotStream — mount starts with clean useChat state", () => {
  it("calls setMessages([]) on mount to avoid stale Chat-instance bleed-through", () => {
    renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-A" }),
    });

    // The mount effect resets the AI SDK's messages array. AI SDK v5
    // caches Chat instances per id, so without this a revisit to a
    // session would render stale pre-refetch messages until hydration
    // catches up.
    expect(mockSetMessages).toHaveBeenCalledWith([]);
  });
});
