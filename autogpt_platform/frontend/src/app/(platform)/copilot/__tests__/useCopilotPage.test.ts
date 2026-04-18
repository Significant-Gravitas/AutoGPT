import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotPage } from "../useCopilotPage";

const mockUseChatSession = vi.fn();
const mockUseCopilotStream = vi.fn();
const mockUseLoadMoreMessages = vi.fn();
const mockPostV2QueuePendingMessage = vi.fn();
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
vi.mock("../store", () => ({
  useCopilotUIStore: () => ({
    sessionToDelete: null,
    setSessionToDelete: vi.fn(),
    isDrawerOpen: false,
    setDrawerOpen: vi.fn(),
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
}));
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useDeleteV2DeleteSession: () => ({ mutate: vi.fn(), isPending: false }),
  useGetV2ListSessions: () => ({ data: undefined, isLoading: false }),
  getGetV2ListSessionsQueryKey: () => ["sessions"],
  getV2GetPendingMessages: vi.fn().mockResolvedValue({
    status: 200,
    data: { count: 0, messages: [] },
  }),
  postV2QueuePendingMessage: (...args: unknown[]) =>
    mockPostV2QueuePendingMessage(...args),
}));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));
vi.mock("@/lib/direct-upload", () => ({
  uploadFileDirect: vi.fn(),
}));
vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => "lg",
}));
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isUserLoading: false, isLoggedIn: true }),
}));
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
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
    historicalDurations: new Map(),
    hasActiveStream: false,
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
    sendMessage: vi.fn(),
    stop: vi.fn(),
    status: "ready",
    error: undefined,
    isReconnecting: false,
    isSyncing: false,
    isUserStoppingRef: { current: false },
    rateLimitMessage: null,
    dismissRateLimit: vi.fn(),
    ...overrides,
  };
}

function makeBaseLoadMore(overrides: Record<string, unknown> = {}) {
  return {
    pagedMessages: [],
    hasMore: false,
    isLoadingMore: false,
    loadMore: vi.fn(),
    ...overrides,
  };
}

describe("useCopilotPage — backward pagination message ordering", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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

describe("useCopilotPage — onSend queue-in-flight path", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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

    expect(mockPostV2QueuePendingMessage).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Please wait to attach files" }),
    );
  });

  it("queues a text-only message via postV2QueuePendingMessage when in flight", async () => {
    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({ status: "streaming" }),
    );
    mockUseLoadMoreMessages.mockReturnValue(makeBaseLoadMore());
    mockPostV2QueuePendingMessage.mockResolvedValue({ status: 202 });

    const { result } = renderHook(() => useCopilotPage());

    await act(async () => {
      await result.current.onSend("follow-up");
    });

    expect(mockPostV2QueuePendingMessage).toHaveBeenCalledWith("sess-1", {
      message: "follow-up",
    });
    // appendChip should have been called, bringing the chip into queuedMessages.
    await waitFor(() => {
      expect(result.current.queuedMessages).toContain("follow-up");
    });
  });

  it("surfaces a toast and rethrows when the queue POST fails", async () => {
    mockUseChatSession.mockReturnValue(makeBaseChatSession());
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({ status: "streaming" }),
    );
    mockUseLoadMoreMessages.mockReturnValue(makeBaseLoadMore());
    mockPostV2QueuePendingMessage.mockRejectedValue(new Error("boom"));

    const { result } = renderHook(() => useCopilotPage());

    await act(async () => {
      await expect(result.current.onSend("follow-up")).rejects.toThrow(/boom/);
    });

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Could not queue message" }),
    );
  });
});
