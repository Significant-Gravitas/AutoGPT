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
  environment: {
    getAGPTServerBaseUrl: () => "http://localhost:8006",
    isServerSide: () => false,
  },
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
vi.mock("../helpers", () => ({
  getCopilotAuthHeaders: vi.fn(async () => ({ Authorization: "Bearer test" })),
  deduplicateMessages: (msgs: UIMessage[]) => msgs,
  parseSessionIDs: (_raw: string | null | undefined) => new Set<string>(),
  extractSendMessageText: (arg: unknown) =>
    arg && typeof arg === "object" && "text" in arg
      ? String((arg as { text: string }).text)
      : String(arg ?? ""),
  hasActiveBackendStream: (result: { data?: unknown }) =>
    mockHasActiveBackendStream(result),
  getLatestAssistantStatusMessage: (messages: UIMessage[]) => {
    const last = messages[messages.length - 1];
    if (last?.role !== "assistant") return null;
    for (let i = last.parts.length - 1; i >= 0; i--) {
      const part = last.parts[i] as UIMessage["parts"][number] & {
        data?: { message?: unknown };
      };
      if (part.type === "data-cursor") continue;
      if (part.type === "data-status") {
        return typeof part.data?.message === "string"
          ? part.data.message
          : null;
      }
      return null;
    }
    return null;
  },
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
}));

// --- ai SDK mock (DefaultChatTransport must be constructible) ---
vi.mock("ai", () => ({
  DefaultChatTransport: vi.fn().mockImplementation(function () {
    return {};
  }),
}));

// --- @ai-sdk/react useChat mock with persistent Chat instances ---
type UseChatOptions = {
  id?: string;
  chat?: { id?: string };
};
const capturedUseChatChats: Array<{ id?: string } | undefined> = [];
const mockResumeStream = vi.fn();
const mockSdkStop = vi.fn();
const mockSdkSendMessage = vi.fn();
const mockSetMessages = vi.fn();
let mockMessages: UIMessage[] = [];
let mockStatus: "ready" | "streaming" | "submitted" | "error" = "ready";

vi.mock("@ai-sdk/react", () => ({
  Chat: class MockChat {
    id: string;
    onFinish?: (args: {
      isDisconnect?: boolean;
      isAbort?: boolean;
    }) => void | Promise<void>;
    onError?: (error: Error) => void;

    constructor(options: {
      id?: string;
      onFinish?: (args: {
        isDisconnect?: boolean;
        isAbort?: boolean;
      }) => void | Promise<void>;
      onError?: (error: Error) => void;
    }) {
      this.id = options.id ?? "";
      this.onFinish = options.onFinish;
      this.onError = options.onError;
    }
  },
  useChat: (opts: UseChatOptions) => {
    capturedUseChatChats.push(opts.chat);
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
import {
  getOrCreateCopilotChatRuntime,
  resetCopilotChatRegistry,
  shouldReloadCopilotChatRuntime,
} from "../copilotChatRegistry";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { RESTORE_STALL_TIMEOUT_MS } from "../restoreConstants";
import { useCopilotUIStore } from "../store";
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
  capturedUseChatChats.length = 0;
  mockResumeStream.mockClear();
  mockSdkStop.mockClear();
  mockSdkSendMessage.mockClear();
  mockSetMessages.mockClear();
  mockToast.mockClear();
  mockHasActiveBackendStream.mockReturnValue(false);
  mockInvalidateQueries.mockClear();
  resetCopilotChatRegistry();
  // Zustand stores are module singletons — wipe per-session coord state so
  // tests don't leak into each other.
  useCopilotStreamStore.getState().resetAll();
});

afterEach(() => {
  vi.useRealTimers();
  cleanup();
});

describe("useCopilotStream — hydration/resume race (SECRT-2242)", () => {
  it("reuses the same session chat runtime across remounts", () => {
    const first = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-1" }),
    });
    first.unmount();

    renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-1" }),
    });

    expect(capturedUseChatChats).toHaveLength(2);
    expect(capturedUseChatChats[0]?.id).toBe("sess-1");
    expect(capturedUseChatChats[1]?.id).toBe("sess-1");
    expect(capturedUseChatChats[0]).toBe(capturedUseChatChats[1]);
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

    mockStatus = "streaming";
    rerender(
      makeArgs({
        hasActiveStream: true,
        hydratedMessages: [],
      }),
    );
    expect(result.current.isRestoringActiveSession).toBe(true);

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

  it("treats a replay status update as enough to leave restore mode", () => {
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

    mockStatus = "streaming";
    mockMessages = [
      {
        id: "a1",
        role: "assistant",
        parts: [
          {
            type: "data-status",
            data: { message: "Analyzing result..." },
          },
        ],
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

  it("keeps the restore latch closed when hydration lands assistant content while status is still ready", () => {
    // Repro for the premature-latch bug: the user reopens a session whose
    // backend stream is still active, hydration lands a partial assistant
    // message persisted mid-stream into rawMessages, but the live SSE
    // GET-resume hasn't produced any bytes yet (status still ``ready``).
    // The latch must NOT fire — otherwise the "Retrieving latest messages"
    // UI is suppressed and the 6 s restore-stall watchdog can't run.
    mockStatus = "ready";
    mockMessages = [
      {
        id: "u1",
        role: "user",
        parts: [{ type: "text", text: "Hi" }],
      } as unknown as UIMessage,
      {
        id: "a1",
        role: "assistant",
        parts: [{ type: "text", text: "Partial response" }],
      } as unknown as UIMessage,
    ];

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

    // Once the live stream actually flips status to ``streaming`` (resume
    // produced its first byte), the latch is allowed to fire.
    mockStatus = "streaming";
    rerender(
      makeArgs({
        hasActiveStream: true,
        hydratedMessages: [],
      }),
    );

    expect(result.current.isRestoringActiveSession).toBe(false);
  });

  it("kicks the reconnect cascade when restore stays stuck for 6 seconds", async () => {
    mockHasActiveBackendStream.mockReturnValue(true);
    const refetchSession = vi.fn(async () => ({
      data: { status: 200, data: { active_stream: { started_at: "now" } } },
    }));

    const { rerender } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: [],
        refetchSession,
      }),
    });

    expect(mockResumeStream).toHaveBeenCalledTimes(1);

    mockStatus = "streaming";
    rerender(
      makeArgs({
        hasActiveStream: true,
        hydratedMessages: [],
        refetchSession,
      }),
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(RESTORE_STALL_TIMEOUT_MS + 1_000);
    });

    expect(refetchSession).toHaveBeenCalled();
    expect(mockResumeStream).toHaveBeenCalledTimes(2);
  });

  it("stores the latest per-session snapshot without wiping it on an empty remount", () => {
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

    const first = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: [userMessage, assistantMessage],
      }),
    });

    expect(
      useCopilotStreamStore.getState().getMessageSnapshot("sess-1"),
    ).toEqual([userMessage, assistantMessage]);

    first.unmount();
    mockMessages = [];

    renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({
        hasActiveStream: true,
        hydratedMessages: [userMessage, assistantMessage],
      }),
    });

    expect(
      useCopilotStreamStore.getState().getMessageSnapshot("sess-1"),
    ).toEqual([userMessage, assistantMessage]);
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
  it("keeps the in-flight fetch alive on unmount", () => {
    const { unmount } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-A" }),
    });

    unmount();

    expect(mockSdkStop).not.toHaveBeenCalled();
  });

  it("reconnect timer does not fire resumeStream after unmount", async () => {
    const { unmount } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-A" }),
    });

    await act(async () => {
      await getOrCreateCopilotChatRuntime("sess-A").onFinish?.({
        isDisconnect: true,
      });
    });

    // Parent would remount us with a new session key — simulate the
    // unmount that happens before the reconnect timer backoff expires.
    unmount();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });

    expect(mockResumeStream).not.toHaveBeenCalled();
  });

  it("marks a background-disconnected session for full reload on reopen", async () => {
    useCopilotStreamStore
      .getState()
      .updateCoord("sess-A", { lastSubmittedMessageText: "hello" });
    useCopilotStreamStore.getState().setMessageSnapshot("sess-A", [
      {
        id: "a1",
        role: "assistant",
        parts: [{ type: "text", text: "partial" }],
      } as unknown as UIMessage,
    ]);

    const firstRuntime = getOrCreateCopilotChatRuntime("sess-A");
    renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-A" }),
    }).unmount();

    await act(async () => {
      await (
        firstRuntime.chat as unknown as {
          onFinish?: (args: { isDisconnect?: boolean }) => Promise<void> | void;
        }
      ).onFinish?.({ isDisconnect: true });
    });

    expect(shouldReloadCopilotChatRuntime("sess-A")).toBe(true);
    expect(
      useCopilotStreamStore.getState().getMessageSnapshot("sess-A"),
    ).toEqual([]);
    expect(
      useCopilotStreamStore.getState().getCoord("sess-A")
        .lastSubmittedMessageText,
    ).toBeNull();

    renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-A" }),
    });

    expect(capturedUseChatChats.at(-1)).not.toBe(firstRuntime.chat);
    expect(shouldReloadCopilotChatRuntime("sess-A")).toBe(false);
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
      await getOrCreateCopilotChatRuntime("sess-1").onFinish?.({
        isDisconnect: true,
      });
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
      await getOrCreateCopilotChatRuntime("sess-1").onFinish?.({
        isDisconnect: true,
      });
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

describe("useCopilotStream — mount preserves session runtime state", () => {
  it("does not clear the chat runtime on mount", () => {
    renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-A" }),
    });

    expect(mockSetMessages).not.toHaveBeenCalledWith([]);
  });
});

describe("useCopilotStream — rate-limit recovery", () => {
  it("restores the unsent text and drops the optimistic user bubble on 429 usage-limit", async () => {
    useCopilotUIStore.getState().setInitialPrompt(null);
    useCopilotStreamStore
      .getState()
      .updateCoord("sess-1", { lastSubmittedMessageText: "recover me" });

    const { result } = renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-1" }),
    });

    await act(async () => {
      await getOrCreateCopilotChatRuntime("sess-1").onError?.(
        new Error(JSON.stringify({ detail: "Daily usage limit exceeded" })),
      );
    });

    expect(useCopilotUIStore.getState().initialPrompt).toBe("recover me");
    expect(result.current.rateLimitMessage).toContain("Daily usage limit");
    expect(
      useCopilotStreamStore.getState().getCoord("sess-1")
        .lastSubmittedMessageText,
    ).toBeNull();

    const dropCall = mockSetMessages.mock.calls.find(([arg]) => {
      if (typeof arg !== "function") return false;
      const prev: UIMessage[] = [
        {
          id: "u1",
          role: "user",
          parts: [{ type: "text", text: "recover me" }],
        } as unknown as UIMessage,
      ];
      const next = (arg as (p: UIMessage[]) => UIMessage[])(prev);
      return next.length === 0;
    });
    expect(dropCall).toBeTruthy();

    // Branch: when the trailing message is NOT a user bubble (e.g. an
    // assistant turn already landed), the rollback updater must leave the
    // list untouched so we don't clobber the assistant reply.
    const updater = mockSetMessages.mock.calls
      .map(([arg]) => arg)
      .find(
        (arg): arg is (p: UIMessage[]) => UIMessage[] =>
          typeof arg === "function",
      );
    expect(updater).toBeDefined();
    const assistantOnly: UIMessage[] = [
      {
        id: "a1",
        role: "assistant",
        parts: [{ type: "text", text: "ok" }],
      } as unknown as UIMessage,
    ];
    expect(updater!(assistantOnly)).toBe(assistantOnly);
    expect(updater!([])).toEqual([]);
  });

  it("skips composer restore on 429 when there is no captured unsent text", async () => {
    useCopilotUIStore.getState().setInitialPrompt(null);
    renderHook((args: Args) => useCopilotStream(args), {
      initialProps: makeArgs({ sessionId: "sess-1" }),
    });

    await act(async () => {
      await getOrCreateCopilotChatRuntime("sess-1").onError?.(
        new Error(JSON.stringify({ detail: "Daily usage limit exceeded" })),
      );
    });

    // No prior sendMessage → coord lastSubmittedMessageText stays null →
    // restore branch must be skipped. The store stays untouched and
    // setMessages is NOT invoked with a rollback updater.
    expect(useCopilotUIStore.getState().initialPrompt).toBeNull();
    const rollbackUpdaterCall = mockSetMessages.mock.calls.find(
      ([arg]) => typeof arg === "function",
    );
    expect(rollbackUpdaterCall).toBeUndefined();
  });

  it("preserves the unsent text in the store when the user is already typing a new draft", async () => {
    useCopilotUIStore.getState().setInitialPrompt(null);
    useCopilotStreamStore
      .getState()
      .updateCoord("sess-1", { lastSubmittedMessageText: "rate-limited" });

    const composer = document.createElement("textarea");
    composer.id = "chat-input";
    composer.value = "user is typing a new draft";
    document.body.appendChild(composer);

    try {
      renderHook((args: Args) => useCopilotStream(args), {
        initialProps: makeArgs({ sessionId: "sess-1" }),
      });

      await act(async () => {
        await getOrCreateCopilotChatRuntime("sess-1").onError?.(
          new Error(JSON.stringify({ detail: "Daily usage limit exceeded" })),
        );
      });

      // Composer non-empty → don't write to the recovery slot, don't clear
      // the per-session unsent buffer. The user keeps their draft and the
      // rate-limited text stays available for a future reload/resume.
      expect(useCopilotUIStore.getState().initialPrompt).toBeNull();
      expect(
        useCopilotStreamStore.getState().getCoord("sess-1")
          .lastSubmittedMessageText,
      ).toBe("rate-limited");
    } finally {
      document.body.removeChild(composer);
    }
  });
});
