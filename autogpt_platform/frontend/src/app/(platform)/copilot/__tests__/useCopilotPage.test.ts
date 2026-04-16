import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotPage } from "../useCopilotPage";

const mockUseChatSession = vi.fn();
const mockUseCopilotStream = vi.fn();
const mockUseLoadMoreMessages = vi.fn();

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
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useDeleteV2DeleteSession: () => ({ mutate: vi.fn(), isPending: false }),
  useGetV2ListSessions: () => ({ data: undefined, isLoading: false }),
  getGetV2ListSessionsQueryKey: () => ["sessions"],
}));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
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
    newestSequence: null,
    forwardPaginated: false,
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
    resetPaged: vi.fn(),
    ...overrides,
  };
}

describe("useCopilotPage — forwardPaginated message ordering", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("prepends pagedMessages before currentMessages when forwardPaginated=false", () => {
    const pagedMsg = { id: "paged", role: "user" };
    const currentMsg = { id: "current", role: "assistant" };
    mockUseChatSession.mockReturnValue(
      makeBaseChatSession({ forwardPaginated: false }),
    );
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

  it("appends pagedMessages after currentMessages when forwardPaginated=true", () => {
    const pagedMsg = { id: "paged", role: "assistant" };
    const currentMsg = { id: "current", role: "user" };
    mockUseChatSession.mockReturnValue(
      makeBaseChatSession({ forwardPaginated: true }),
    );
    mockUseCopilotStream.mockReturnValue(
      makeBaseCopilotStream({ messages: [currentMsg] }),
    );
    mockUseLoadMoreMessages.mockReturnValue(
      makeBaseLoadMore({ pagedMessages: [pagedMsg] }),
    );

    const { result } = renderHook(() => useCopilotPage());

    // Forward: currentMessages (beginning of session) come first
    expect(result.current.messages[0]).toEqual(currentMsg);
    expect(result.current.messages[1]).toEqual(pagedMsg);
  });

  it("calls resetPaged when forwardPaginated transitions false→true with paged messages", async () => {
    const mockResetPaged = vi.fn();
    const pagedMsg = { id: "paged", role: "user" };

    mockUseChatSession.mockReturnValue(
      makeBaseChatSession({ forwardPaginated: false }),
    );
    mockUseCopilotStream.mockReturnValue(makeBaseCopilotStream());
    mockUseLoadMoreMessages.mockReturnValue(
      makeBaseLoadMore({
        pagedMessages: [pagedMsg],
        resetPaged: mockResetPaged,
      }),
    );

    const { rerender } = renderHook(() => useCopilotPage());

    // Simulate session completing — forwardPaginated flips to true
    mockUseChatSession.mockReturnValue(
      makeBaseChatSession({ forwardPaginated: true }),
    );

    act(() => {
      rerender();
    });

    await waitFor(() => {
      expect(mockResetPaged).toHaveBeenCalled();
    });
  });

  it("does not call resetPaged when forwardPaginated is already true on mount", () => {
    const mockResetPaged = vi.fn();
    mockUseChatSession.mockReturnValue(
      makeBaseChatSession({ forwardPaginated: true }),
    );
    mockUseCopilotStream.mockReturnValue(makeBaseCopilotStream());
    mockUseLoadMoreMessages.mockReturnValue(
      makeBaseLoadMore({ pagedMessages: [], resetPaged: mockResetPaged }),
    );

    renderHook(() => useCopilotPage());

    expect(mockResetPaged).not.toHaveBeenCalled();
  });
});
