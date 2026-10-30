import { act, renderHook } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The page hook reads three fields of the copilot UI store. A write to any
// other field (a panel resize writes `contextPanelWidth` every frame of a
// drag) must not re-render it, because it re-renders the whole chat host.

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

import { DEFAULT_PANEL_WIDTH, useCopilotUIStore } from "../store";
import { useCopilotPage } from "../useCopilotPage";

const initialState = useCopilotUIStore.getState();

beforeEach(() => {
  streamState.status = "ready";
  sendNewMessage.mockResolvedValue(undefined);
  useCopilotUIStore.setState({
    contextPanelWidth: DEFAULT_PANEL_WIDTH,
    copilotLlmModel: "standard",
    isDryRun: false,
    initialPrompt: null,
  });
});

afterEach(() => {
  useCopilotUIStore.setState(initialState);
  vi.clearAllMocks();
  window.sessionStorage.clear();
  window.localStorage.clear();
});

function renderCountedPage() {
  const counter = { renders: 0 };
  const view = renderHook(() => {
    counter.renders++;
    return useCopilotPage();
  });
  return { ...view, counter };
}

describe("useCopilotPage — copilot UI store reads", () => {
  it("does not re-render for writes to fields it does not read", async () => {
    const { counter } = renderCountedPage();
    await act(async () => {});
    const before = counter.renders;

    act(() => {
      const store = useCopilotUIStore.getState();
      store.setContextPanelWidth(DEFAULT_PANEL_WIDTH + 40);
      store.setArtifactPanelWidth(700);
      store.setDrawerOpen(true);
      store.setSearchOpen(true);
    });

    expect(counter.renders).toBe(before);
  });

  it("still re-renders when the model or dry-run setting changes", async () => {
    const { counter } = renderCountedPage();
    await act(async () => {});
    const before = counter.renders;

    act(() => useCopilotUIStore.getState().setCopilotLlmModel("advanced"));
    expect(counter.renders).toBeGreaterThan(before);

    const afterModel = counter.renders;
    act(() => useCopilotUIStore.getState().setIsDryRun(true));
    expect(counter.renders).toBeGreaterThan(afterModel);
  });
});
