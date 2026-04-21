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
  resolveInProgressTools: (msgs: UIMessage[]) => msgs,
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
  onFinish?: (args: OnFinishArgs) => void | Promise<void>;
  onError?: (e: Error) => void;
};
let capturedUseChatOptions: UseChatOptions | null = null;
const mockResumeStream = vi.fn();
const mockSdkStop = vi.fn();
const mockSdkSendMessage = vi.fn();
const mockSetMessages = vi.fn();
let mockMessages: UIMessage[] = [];
let mockStatus: "ready" | "streaming" | "submitted" | "error" = "ready";

vi.mock("@ai-sdk/react", () => ({
  useChat: (opts: UseChatOptions) => {
    capturedUseChatOptions = opts;
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

describe("useCopilotStream — cursor tracking for incremental resume", () => {
  it("records the latest data-cursor chunkId on the per-session coord", () => {
    const assistantWithCursor: UIMessage = {
      id: "assistant-1",
      role: "assistant",
      parts: [
        { type: "text", text: "hi", state: "done" },
        {
          type: "data-cursor",
          data: { chunkId: "1700000000000-0" },
        } as unknown as UIMessage["parts"][number],
        { type: "text", text: "there", state: "done" },
        {
          type: "data-cursor",
          data: { chunkId: "1700000000001-0" },
        } as unknown as UIMessage["parts"][number],
      ],
    };
    mockMessages = [assistantWithCursor];

    renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ hasActiveStream: false }),
    });

    expect(
      useCopilotStreamStore.getState().getCoord("sess-1").lastChunkId,
    ).toBe("1700000000001-0");
  });

  it("preserves cursor across a simulated remount (persists per session)", () => {
    // First mount populates the cursor.
    const assistantWithCursor: UIMessage = {
      id: "assistant-1",
      role: "assistant",
      parts: [
        {
          type: "data-cursor",
          data: { chunkId: "1700000000005-0" },
        } as unknown as UIMessage["parts"][number],
      ],
    };
    mockMessages = [assistantWithCursor];

    const { unmount } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-A", hasActiveStream: false }),
    });

    unmount();

    // Remount for the same session — cursor must still be there so the
    // transport's prepareReconnectToStreamRequest can pass it on resume.
    expect(
      useCopilotStreamStore.getState().getCoord("sess-A").lastChunkId,
    ).toBe("1700000000005-0");
  });
});

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
