import { renderHook, waitFor } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The page's send handler is one function for the chat host's lifetime, so
// the chat actions context and the composer keep their props while a reply
// streams in; it still runs with the state of the latest render.

const streamState = vi.hoisted(() => ({
  status: "ready" as "ready" | "submitted" | "streaming" | "error",
}));

const sendNewMessage = vi.hoisted(() => vi.fn());
const completeGreeting = vi.hoisted(() => vi.fn());
const queueFollowUpMessage = vi.hoisted(() => vi.fn());
const queueMessage = vi.hoisted(() => vi.fn());
const setPendingFileParts = vi.hoisted(() => vi.fn());

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "chat-mode-option",
  },
  useGetFlag: () => false,
}));

vi.mock("nuqs", () => ({
  parseAsString: {},
  useQueryState: () => [null, vi.fn()],
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

vi.mock("@/app/api/__generated__/endpoints/brain-dump/brain-dump", () => ({
  useCompleteBrainDumpGreeting: () => ({ mutate: completeGreeting }),
}));

vi.mock("../useExpertMap", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../useExpertMap")>();
  return {
    ...actual,
    useExpertMap: () => ({
      expertsById: new Map(),
      isLoadingExperts: false,
      hasExpertsSettled: true,
      hasExpertsErrored: false,
    }),
  };
});

vi.mock("../useChatSession", () => ({
  useChatSession: () => ({
    sessionId: "session-1",
    setSessionId: vi.fn(),
    sessionLlmAuthProvider: "platform",
    sessionExpertId: null,
    isAdoptingExpertSession: false,
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
    sessionChatStatus: "idle",
  }),
}));

vi.mock("../useCopilotStream", () => ({
  useCopilotStream: () => ({
    messages: [],
    setMessages: vi.fn(),
    sendMessage: vi.fn(),
    stop: vi.fn(),
    status: streamState.status,
    error: undefined,
    isReconnecting: false,
    isFinishProbing: false,
    isRestoringActiveSession: false,
    isSyncing: false,
    isUserStoppingRef: { current: false },
    isUserStopping: false,
    rateLimitMessage: null,
    dismissRateLimit: vi.fn(),
  }),
}));

vi.mock("../useSendMessage", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../useSendMessage")>();
  return {
    ...actual,
    useSendMessage: () => ({
      onSend: sendNewMessage,
      isUploadingFiles: false,
      setPendingFileParts,
    }),
  };
});

vi.mock("../useLoadMoreMessages", () => ({
  useLoadMoreMessages: () => ({
    pagedMessages: [],
    pagedTurnStats: new Map(),
    hasMore: false,
    isLoadingMore: false,
    loadMore: vi.fn(),
  }),
}));

vi.mock("../useCopilotPendingChips", () => ({
  useCopilotPendingChips: () => ({ queuedMessages: [], queueMessage }),
}));

vi.mock("../useCopilotNotifications", () => ({
  useCopilotNotifications: () => undefined,
}));
vi.mock("../useSessionTitlePoll", () => ({
  useSessionTitlePoll: () => undefined,
}));
vi.mock("../useExpertKickoff", () => ({
  useExpertKickoff: () => ({ isKickoffStarting: false }),
}));
vi.mock("../helpers/queueFollowUpMessage", () => ({ queueFollowUpMessage }));

import { useCopilotUIStore } from "../store";
import { useCopilotPage } from "../useCopilotPage";

beforeEach(() => {
  streamState.status = "ready";
  sendNewMessage.mockResolvedValue(undefined);
  queueFollowUpMessage.mockResolvedValue(undefined);
  useCopilotUIStore.getState().setInitialPrompt(null);
});

afterEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
  window.localStorage.clear();
});

describe("useCopilotPage — stable onSend", () => {
  it("keeps one onSend and onEnqueue across re-renders", () => {
    const { result, rerender } = renderHook(() => useCopilotPage());
    const { onSend, onEnqueue } = result.current;
    expect(onEnqueue).toBe(onSend);

    streamState.status = "streaming";
    rerender();
    streamState.status = "ready";
    rerender();

    expect(result.current.onSend).toBe(onSend);
    expect(result.current.onEnqueue).toBe(onSend);
  });

  it("completes the greeting through a handler kept since mount", async () => {
    const { result, rerender } = renderHook(() => useCopilotPage());
    const keptOnSend = result.current.onSend;
    rerender();
    await keptOnSend("hello");

    expect(completeGreeting).toHaveBeenCalledTimes(1);
    expect(sendNewMessage).toHaveBeenCalledWith(
      "hello",
      undefined,
      undefined,
      undefined,
    );
  });

  it("queues a follow-up through the kept handler once a turn is running", async () => {
    const { result, rerender } = renderHook(() => useCopilotPage());
    const keptOnSend = result.current.onSend;

    streamState.status = "streaming";
    rerender();
    await keptOnSend("and also this");

    expect(queueFollowUpMessage).toHaveBeenCalledWith(
      "session-1",
      "and also this",
    );
    expect(queueMessage).toHaveBeenCalledWith("and also this");
    expect(sendNewMessage).not.toHaveBeenCalled();
  });

  it("still auto-submits a workflow import prompt that lands after mount", async () => {
    const { rerender } = renderHook(() => useCopilotPage());
    expect(sendNewMessage).not.toHaveBeenCalled();

    window.sessionStorage.setItem("importWorkflowPrompt", "Import this");
    window.sessionStorage.setItem("importWorkflowAutosubmit", "true");
    rerender();

    await waitFor(() =>
      expect(sendNewMessage).toHaveBeenCalledWith(
        "Import this",
        undefined,
        undefined,
        undefined,
      ),
    );
  });
});
