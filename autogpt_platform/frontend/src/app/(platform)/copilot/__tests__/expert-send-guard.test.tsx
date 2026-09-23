import { renderHook, waitFor } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

// The guard under test is `if (isExpertSendLocked) return;` inside
// useCopilotPage.onSend — the LAST line of defence, behind ChatContainer's
// guardedOnSend wrapper. Deleting it would leave every ChatContainer test
// green while the hook itself silently sent to a fired expert, so it needs
// coverage that does not go through the component.

const expertMapState = vi.hoisted(() => ({
  expertsById: new Map<string, unknown>(),
  hasExpertsSettled: true,
  hasExpertsErrored: false,
}));

const sendNewMessage = vi.hoisted(() => vi.fn());
const queueFollowUpMessage = vi.hoisted(() => vi.fn());

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "chat-mode-option",
    HIRE_EXPERTS: "hire-experts",
    ONBOARDING_BRAIN_DUMP: "onboarding-brain-dump",
  },
  useGetFlag: (flag: string) => flag === "hire-experts",
}));

vi.mock("nuqs", () => ({
  parseAsString: {},
  useQueryState: () => ["expert-maria", vi.fn()],
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

vi.mock("@/app/api/__generated__/endpoints/brain-dump/brain-dump", () => ({
  useCompleteBrainDumpGreeting: () => ({ mutate: vi.fn() }),
}));

vi.mock("../useExpertMap", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../useExpertMap")>();
  return { ...actual, useExpertMap: () => expertMapState };
});

vi.mock("../useChatSession", () => ({
  useChatSession: () => ({
    sessionId: "session-1",
    setSessionId: vi.fn(),
    sessionLlmAuthProvider: "platform",
    sessionExpertId: "expert-maria",
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
    status: "ready",
    error: undefined,
    isReconnecting: false,
    isRestoringActiveSession: false,
    isSyncing: false,
    isUserStoppingRef: { current: false },
    isUserStopping: false,
    rateLimitMessage: null,
    dismissRateLimit: vi.fn(),
  }),
}));

vi.mock("../useSendMessage", () => ({
  useSendMessage: () => ({
    onSend: sendNewMessage,
    isUploadingFiles: false,
    setPendingFileParts: vi.fn(),
  }),
}));

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
  useCopilotPendingChips: () => ({ queuedMessages: [], queueMessage: vi.fn() }),
}));

vi.mock("../useCopilotNotifications", () => ({
  useCopilotNotifications: () => undefined,
}));
vi.mock("../useSessionTitlePoll", () => ({
  useSessionTitlePoll: () => undefined,
}));
vi.mock("../useWorkflowImportAutoSubmit", () => ({
  useWorkflowImportAutoSubmit: () => undefined,
}));
vi.mock("../useExpertKickoff", () => ({
  useExpertKickoff: () => ({ isKickoffStarting: false }),
}));
vi.mock("../helpers/queueFollowUpMessage", () => ({ queueFollowUpMessage }));

import { useCopilotPage } from "../useCopilotPage";

function setExpert(identity: {
  isArchived: boolean;
  readOnlyReason: "fired" | "unavailable" | "unknown" | null;
}) {
  expertMapState.expertsById = new Map([
    [
      "expert-maria",
      {
        id: "expert-maria",
        name: "Maria",
        avatarUrl: null,
        role: "Marketing Strategist",
        ...identity,
      },
    ],
  ]);
  expertMapState.hasExpertsSettled = true;
  expertMapState.hasExpertsErrored = false;
}

afterEach(() => {
  sendNewMessage.mockReset();
  queueFollowUpMessage.mockReset();
});

describe("useCopilotPage — expert send guard", () => {
  it("drops a send when the session's expert has been fired", async () => {
    setExpert({ isArchived: true, readOnlyReason: "fired" });

    const { result } = renderHook(() => useCopilotPage());
    await result.current.onSend("are you still there?");

    expect(sendNewMessage).not.toHaveBeenCalled();
    expect(queueFollowUpMessage).not.toHaveBeenCalled();
  });

  it("drops a send while the expert identity is still unresolved", async () => {
    setExpert({ isArchived: false, readOnlyReason: null });
    expertMapState.hasExpertsSettled = false;

    const { result } = renderHook(() => useCopilotPage());
    await result.current.onSend("hello");

    expect(sendNewMessage).not.toHaveBeenCalled();
  });

  it("still sends for an active expert", async () => {
    setExpert({ isArchived: false, readOnlyReason: null });

    const { result } = renderHook(() => useCopilotPage());
    await result.current.onSend("hello");

    await waitFor(() => expect(sendNewMessage).toHaveBeenCalledTimes(1));
  });
});
