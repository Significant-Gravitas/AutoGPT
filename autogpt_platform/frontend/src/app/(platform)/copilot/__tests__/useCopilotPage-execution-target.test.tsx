import { act, renderHook } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// A Local PC target picked while the rollout flag was on can't be sent once
// it's off: the backend refuses the create, and the picker that could switch
// back to Cloud is hidden, so every send failed.

const flags = vi.hoisted(() => ({ localPC: true }));
const chatSessionCalls = vi.hoisted(
  () => [] as { executionTarget?: unknown }[],
);

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
    LOCAL_PC_EXECUTOR: "local-pc-executor",
  },
  useGetFlag: (flag: string) =>
    flag === "local-pc-executor" ? flags.localPC : false,
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
  useChatSession: (args: { executionTarget?: unknown }) => {
    chatSessionCalls.push(args);
    return {
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
    };
  },
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

const initialState = useCopilotUIStore.getState();

const LOCAL_TARGET = {
  kind: "local" as const,
  machineID: "machine-1",
  machineLabel: "Workstation",
  connectionID: "connection-1",
  browseID: "browse-1",
  directoryRef: "directory-1",
  displayPath: "/Users/ada/project",
};

beforeEach(() => {
  streamState.status = "ready";
  sendNewMessage.mockResolvedValue(undefined);
  flags.localPC = true;
  chatSessionCalls.length = 0;
});

afterEach(() => {
  useCopilotUIStore.setState(initialState);
  vi.clearAllMocks();
  window.sessionStorage.clear();
  window.localStorage.clear();
});

describe("useCopilotPage — new chat execution target", () => {
  it("sends a picked Local PC target while Local PC is on", async () => {
    useCopilotUIStore.getState().setNewChatExecutionTarget(LOCAL_TARGET);

    renderHook(() => useCopilotPage());
    await act(async () => {});

    expect(chatSessionCalls.at(-1)?.executionTarget).toEqual(LOCAL_TARGET);
  });

  it("holds a picked Local PC target back once Local PC is off", async () => {
    useCopilotUIStore.getState().setNewChatExecutionTarget(LOCAL_TARGET);
    flags.localPC = false;

    renderHook(() => useCopilotPage());
    await act(async () => {});

    expect(chatSessionCalls.at(-1)?.executionTarget).toBeUndefined();
  });
});
