import { act, renderHook, waitFor } from "@testing-library/react";
import type { UIMessage } from "ai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useCopilotPage } from "../useCopilotPage";

const mockUseChatSession = vi.fn();
const mockUseCopilotStream = vi.fn();
const mockUseLoadMoreMessages = vi.fn();
const mockQueueFollowUpMessage = vi.fn();
const mockSendNewMessage = vi.fn();
const mockToast = vi.fn();

vi.mock("../useChatSession", () => ({
  useChatSession: (...args: unknown[]) => mockUseChatSession(...args),
}));
vi.mock("../useCopilotStream", () => ({
  useCopilotStream: (...args: unknown[]) => mockUseCopilotStream(...args),
}));
vi.mock("../useLoadMoreMessages", () => ({
  useLoadMoreMessages: (...args: unknown[]) => mockUseLoadMoreMessages(...args),
}));
vi.mock("../useCopilotNotifications", () => ({
  useCopilotNotifications: () => undefined,
}));
vi.mock("../useWorkflowImportAutoSubmit", () => ({
  useWorkflowImportAutoSubmit: () => undefined,
}));
vi.mock("../useSendMessage", () => ({
  useSendMessage: () => ({
    onSend: (...args: unknown[]) => mockSendNewMessage(...args),
    isUploadingFiles: false,
    setPendingFileParts: vi.fn(),
  }),
}));
vi.mock("../useSessionTitlePoll", () => ({
  useSessionTitlePoll: () => undefined,
}));
vi.mock("../store", () => ({
  useCopilotUIStore: () => ({
    copilotChatMode: "chat",
    copilotLlmModel: null,
    isDryRun: false,
  }),
}));
vi.mock("../helpers/convertChatSessionToUiMessages", () => ({
  concatWithAssistantMerge: (a: unknown[], b: unknown[]) => [...a, ...b],
}));
vi.mock("../helpers", () => ({
  deduplicateMessages: (msgs: unknown[]) => msgs,
  getLatestAssistantStatusMessage: (
    messages: Array<{ role?: string; parts?: unknown[] }>,
  ) => {
    const last = messages[messages.length - 1] as
      | {
          role?: string;
          parts?: Array<{ type?: string; data?: { message?: unknown } }>;
        }
      | undefined;
    if (last?.role !== "assistant") return null;
    for (let i = (last.parts?.length ?? 0) - 1; i >= 0; i--) {
      const part = last.parts?.[i];
      if (part?.type === "data-cursor") continue;
      if (part?.type === "data-status") {
        return typeof part.data?.message === "string"
          ? part.data.message
          : null;
      }
      return null;
    }
    return null;
  },
}));
vi.mock("../helpers/queueFollowUpMessage", () => ({
  queueFollowUpMessage: (...args: unknown[]) =>
    mockQueueFollowUpMessage(...args),
}));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isUserLoading: false, isLoggedIn: true }),
}));
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { CHAT_MODE_OPTION: "CHAT_MODE_OPTION" },
  useGetFlag: () => false,
}));

function makeBaseChatSession(overrides: Record<string, unknown> = {}) {
  return {
    sessionId: "sess-1",
    setSessionId: vi.fn(),
    hydratedMessages: [],
    rawSessionMessages: [],
    historicalTurnStats: new Map(),
    hasActiveStream: false,
    activeStreamStartedAt: null,
    hasMoreMessages: false,
    oldestSequence: null,
    isLoadingSession: false,
    isSessionError: false,
    createSession: vi.fn(),
    isCreatingSession: false,
    refetchSession: vi.fn(),
    sessionDryRun: false,
    ...overrides,
  };
}

function makeBaseCopilotStream(overrides: Record<string, unknown> = {}) {
  return {
    messages: [],
    setMessages: vi.fn(),
    sendMessage: vi.fn(),
    stop: vi.fn(),
    status: "ready",
    error: undefined,
    isReconnecting: false,
    isRestoringActiveSession: false,
    restoreStatusMessage: null,
    isSyncing: false,
    isUserStopping: false,
    isUserStoppingRef: { current: false },
    rateLimitMessage: null,
    dismissRateLimit: vi.fn(),
    ...overrides,
  };
}

function makeBaseLoadMore(overrides: Record<string, unknown> = {}) {
  return {
    pagedMessages: [],
    pagedTurnStats: new Map(),
    hasMore: false,
    isLoadingMore: false,
    loadMore: vi.fn(),
    ...overrides,
  };
}

describe("useCopilotPage — backward pagination message ordering", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCopilotStreamStore.getState().resetAll();
  });

  it("prepends pagedMessages before currentMessages", () => {
    const pagedMsg = { id: "paged", role: "user" };
    const currentMsg = { id: "current", role: "assistant" };
    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({ messages: [currentMsg] }),
    );
    mockUseLoadMoreMessages.mockReturnValue(
      makeBaseLoadMore({ pagedMessages: [pagedMsg] }),
    );

    const { result } = renderHook(() => useCopilotPage());

    // Backward: pagedMessages (older) come first
    expect(result.current.messages[0]).toEqual(pagedMsg);
    expect(result.current.messages[1]).toEqual(currentMsg);
  });
});

describe("useCopilotPage — active session restore visibility", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCopilotStreamStore.getState().resetAll();
  });

  it("hides the trailing assistant tail until an active session is resumed", () => {
    const userMessage = {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Tell me more" }],
    };
    const assistantMessage = {
      id: "assistant-1",
      role: "assistant",
      parts: [{ type: "reasoning", text: "Working on it" }],
    };

    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({
        messages: [userMessage, assistantMessage],
        isRestoringActiveSession: true,
      }),
    );
    mockUseLoadMoreMessages.mockReturnValue(makeBaseLoadMore());

    const { result } = renderHook(() => useCopilotPage());

    expect(result.current.messages).toEqual([userMessage]);
    expect(result.current.isRestoringActiveSession).toBe(true);
  });

  it("preserves the hidden replay status message while restore trims the assistant tail", () => {
    const userMessage = {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Tell me more" }],
    };
    const assistantMessage = {
      id: "assistant-1",
      role: "assistant",
      parts: [
        {
          type: "data-status",
          data: { message: "Contacting the model..." },
        },
      ],
    };

    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({
        messages: [userMessage, assistantMessage],
        isRestoringActiveSession: true,
      }),
    );
    mockUseLoadMoreMessages.mockReturnValue(makeBaseLoadMore());

    const { result } = renderHook(() => useCopilotPage());

    expect(result.current.messages).toEqual([userMessage]);
    expect(result.current.restoreStatusMessage).toBe("Contacting the model...");
  });

  it("prefers the cached live snapshot while restore is still reconnecting", () => {
    const userMessage: UIMessage = {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Tell me more" }],
    };
    const cachedAssistantMessage: UIMessage = {
      id: "assistant-1",
      role: "assistant",
      parts: [{ type: "reasoning", text: "Working on it" }],
    };

    useCopilotStreamStore
      .getState()
      .setMessageSnapshot("sess-1", [userMessage, cachedAssistantMessage]);

    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({
        messages: [userMessage],
        isRestoringActiveSession: true,
      }),
    );
    mockUseLoadMoreMessages.mockReturnValue(makeBaseLoadMore());

    const { result } = renderHook(() => useCopilotPage());

    expect(result.current.messages).toEqual([
      userMessage,
      cachedAssistantMessage,
    ]);
  });
});

describe("useCopilotPage — onEnqueue and queuedMessages", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCopilotStreamStore.getState().resetAll();
  });

  it("exposes onEnqueue (delegating to onSend) and queuedMessages", () => {
    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(makeBaseCopilotStream());
    mockUseLoadMoreMessages.mockReturnValue(makeBaseLoadMore());

    const { result } = renderHook(() => useCopilotPage());

    expect(typeof result.current.onEnqueue).toBe("function");
    expect(Array.isArray(result.current.queuedMessages)).toBe(true);
  });
});

describe("useCopilotPage — turnStats map merge across pages", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("merges historical (current-page) over paged (older) stats; current-page wins on overlap", () => {
    const pagedTurnStats = new Map([
      ["older", { durationMs: 1000, createdAt: "2026-04-20T10:00:00Z" }],
      ["shared", { durationMs: 2000, createdAt: "2026-04-20T10:00:00Z" }],
    ]);
    const historicalTurnStats = new Map([
      ["current", { durationMs: 3000, createdAt: "2026-04-23T08:32:09Z" }],
      ["shared", { durationMs: 4000, createdAt: "2026-04-23T08:32:09Z" }],
    ]);

    mockUseChatSession.mockReturnValue(
      makeBaseChatSession({ historicalTurnStats }),
    );
    mockUseCopilotStream.mockReturnValue(makeBaseCopilotStream());
    mockUseLoadMoreMessages.mockReturnValue(
      makeBaseLoadMore({ pagedTurnStats }),
    );

    const { result } = renderHook(() => useCopilotPage());
    const stats = result.current.turnStats;

    expect(stats.get("older")).toEqual({
      durationMs: 1000,
      createdAt: "2026-04-20T10:00:00Z",
    });
    expect(stats.get("current")).toEqual({
      durationMs: 3000,
      createdAt: "2026-04-23T08:32:09Z",
    });
    // Current-page wins on shared keys.
    expect(stats.get("shared")).toEqual({
      durationMs: 4000,
      createdAt: "2026-04-23T08:32:09Z",
    });
  });
});

describe("useCopilotPage — onSend queue-in-flight path", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useCopilotStreamStore.getState().resetAll();
  });

  it("rejects attaching files while a turn is in flight", async () => {
    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({ status: "streaming" }),
    );
    mockUseLoadMoreMessages.mockReturnValue(makeBaseLoadMore());

    const { result } = renderHook(() => useCopilotPage());

    const bigFile = new File(["x"], "doc.txt", { type: "text/plain" });
    await act(async () => {
      await result.current.onSend("hello", [bigFile]);
    });

    expect(mockQueueFollowUpMessage).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Please wait to attach files" }),
    );
  });

  it("queues a text-only message via queueFollowUpMessage when in flight", async () => {
    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({ status: "streaming" }),
    );
    mockUseLoadMoreMessages.mockReturnValue(makeBaseLoadMore());
    mockQueueFollowUpMessage.mockResolvedValue({
      buffer_length: 1,
      max_buffer_length: 10,
      turn_in_flight: true,
    });

    const { result } = renderHook(() => useCopilotPage());

    await act(async () => {
      await result.current.onSend("follow-up");
    });

    expect(mockQueueFollowUpMessage).toHaveBeenCalledWith(
      "sess-1",
      "follow-up",
    );
    // appendChip should have been called, bringing the chip into queuedMessages.
    await waitFor(() => {
      expect(result.current.queuedMessages).toContain("follow-up");
    });
  });

  it("falls back to a normal send when the active turn has already ended", async () => {
    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({ status: "streaming" }),
    );
    mockUseLoadMoreMessages.mockReturnValue(makeBaseLoadMore());
    const notActiveError = new Error("no active turn");
    notActiveError.name = "QueueFollowUpNotActiveError";
    mockQueueFollowUpMessage.mockRejectedValue(notActiveError);

    const { result } = renderHook(() => useCopilotPage());

    await act(async () => {
      await result.current.onSend("follow-up");
    });

    expect(mockQueueFollowUpMessage).toHaveBeenCalledWith(
      "sess-1",
      "follow-up",
    );
    expect(mockSendNewMessage).toHaveBeenCalledWith("follow-up", undefined);
    expect(result.current.queuedMessages).not.toContain("follow-up");
    expect(mockToast).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "Could not queue message" }),
    );
  });

  it("treats a user-stopped turn as no longer in flight for new sends", async () => {
    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({
        status: "streaming",
        isUserStopping: true,
      }),
    );
    mockUseLoadMoreMessages.mockReturnValue(makeBaseLoadMore());

    const { result } = renderHook(() => useCopilotPage());

    await act(async () => {
      await result.current.onSend("follow-up");
    });

    expect(mockQueueFollowUpMessage).not.toHaveBeenCalled();
    expect(mockSendNewMessage).toHaveBeenCalledWith("follow-up", undefined);
  });

  it("surfaces a toast and rethrows when the queue POST fails", async () => {
    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({ status: "streaming" }),
    );
    mockUseLoadMoreMessages.mockReturnValue(makeBaseLoadMore());
    mockQueueFollowUpMessage.mockRejectedValue(new Error("boom"));

    const { result } = renderHook(() => useCopilotPage());

    await act(async () => {
      await expect(result.current.onSend("follow-up")).rejects.toThrow(/boom/);
    });

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Could not queue message" }),
    );
  });
});
