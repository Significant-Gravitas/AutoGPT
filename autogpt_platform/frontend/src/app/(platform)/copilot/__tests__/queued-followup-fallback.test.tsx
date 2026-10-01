import { act, renderHook } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// A follow-up sent while the page still thinks a turn is in flight goes to
// the pending endpoint. When the backend's turn is already over it answers
// 409 and the page falls back to a normal send — which must not start while
// this tab is still drawing the answer, and must never reach another chat.

const streamState = vi.hoisted(() => ({
  status: "streaming" as "submitted" | "streaming" | "ready" | "error",
  isFinishProbing: false,
  isReconnecting: false,
}));
const sessionState = vi.hoisted(() => ({ sessionId: "session-1" as string }));

const sendNewMessage = vi.hoisted(() => vi.fn());
const queueFollowUpMessage = vi.hoisted(() => vi.fn());
const toast = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({ toast }));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "chat-mode-option",
    HIRE_EXPERTS: "hire-experts",
    ONBOARDING_BRAIN_DUMP: "onboarding-brain-dump",
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
  useCompleteBrainDumpGreeting: () => ({ mutate: vi.fn() }),
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
    sessionId: sessionState.sessionId,
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
    isReconnecting: streamState.isReconnecting,
    isFinishProbing: streamState.isFinishProbing,
    isRestoringActiveSession: false,
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
vi.mock("../helpers/queueFollowUpMessage", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("../helpers/queueFollowUpMessage")>();
  return { ...actual, queueFollowUpMessage };
});

import { QueueFollowUpNotActiveError } from "../helpers/queueFollowUpMessage";
import { useCopilotPage } from "../useCopilotPage";

function setStream(next: Partial<typeof streamState>) {
  Object.assign(streamState, next);
}

async function flushMicrotasks() {
  await act(async () => {
    await Promise.resolve();
  });
}

beforeEach(() => {
  setStream({
    status: "streaming",
    isFinishProbing: false,
    isReconnecting: false,
  });
  sessionState.sessionId = "session-1";
  sendNewMessage.mockResolvedValue(undefined);
  queueFollowUpMessage.mockRejectedValue(new QueueFollowUpNotActiveError());
});

afterEach(() => {
  sendNewMessage.mockReset();
  queueFollowUpMessage.mockReset();
  toast.mockReset();
});

describe("useCopilotPage — follow-up after the backend's turn ended", () => {
  it("waits for this tab's stream, probe and reconnect to settle before sending", async () => {
    const view = renderHook(() => useCopilotPage());

    let accepted = false;
    const send = view.result.current
      .onSend("After the essay, reply with PINEAPPLE")
      .then(() => (accepted = true));
    await flushMicrotasks();
    expect(queueFollowUpMessage).toHaveBeenCalledWith(
      "session-1",
      "After the essay, reply with PINEAPPLE",
    );
    expect(sendNewMessage).not.toHaveBeenCalled();

    // The answer finished typing out, but the post-finish probe is still
    // deciding whether the backend is continuing.
    setStream({ status: "ready", isFinishProbing: true });
    view.rerender();
    await flushMicrotasks();
    expect(sendNewMessage).not.toHaveBeenCalled();

    // The probe found a live backend stream and a reconnect is scheduled.
    setStream({ isFinishProbing: false, isReconnecting: true });
    view.rerender();
    await flushMicrotasks();
    expect(sendNewMessage).not.toHaveBeenCalled();

    setStream({ isReconnecting: false });
    view.rerender();
    await flushMicrotasks();
    await send;
    expect(sendNewMessage).toHaveBeenCalledTimes(1);
    expect(sendNewMessage).toHaveBeenCalledWith(
      "After the essay, reply with PINEAPPLE",
      undefined,
      undefined,
      undefined,
    );
    expect(accepted).toBe(true);
  });

  it("sends straight away when the stream had already settled by the time the 409 arrived", async () => {
    const view = renderHook(() => useCopilotPage());
    let release: () => void = () => undefined;
    queueFollowUpMessage.mockImplementation(
      () =>
        new Promise((_, reject) => {
          release = () => reject(new QueueFollowUpNotActiveError());
        }),
    );

    const send = view.result.current.onSend("follow-up");
    await flushMicrotasks();
    setStream({ status: "ready" });
    view.rerender();

    await act(async () => {
      release();
    });
    await send;
    expect(sendNewMessage).toHaveBeenCalledTimes(1);
  });

  it("drops the follow-up when the user switches chat before the stream settles", async () => {
    const view = renderHook(() => useCopilotPage());

    const send = view.result.current.onSend("follow-up");
    await flushMicrotasks();
    expect(sendNewMessage).not.toHaveBeenCalled();

    sessionState.sessionId = "session-2";
    view.rerender();
    await send;
    expect(sendNewMessage).not.toHaveBeenCalled();

    // The other chat settling must not send the first chat's follow-up.
    setStream({ status: "ready" });
    view.rerender();
    await flushMicrotasks();
    expect(sendNewMessage).not.toHaveBeenCalled();
  });

  it("drops the follow-up when the chat unmounts before the stream settles", async () => {
    const view = renderHook(() => useCopilotPage());

    const send = view.result.current.onSend("follow-up");
    await flushMicrotasks();
    expect(sendNewMessage).not.toHaveBeenCalled();

    view.unmount();
    await send;
    expect(sendNewMessage).not.toHaveBeenCalled();
  });

  it("reports a follow-up whose dispatched send fails", async () => {
    const view = renderHook(() => useCopilotPage());
    sendNewMessage.mockRejectedValue(new Error("network down"));

    const send = view.result.current.onSend("follow-up");
    await flushMicrotasks();
    setStream({ status: "ready" });
    view.rerender();
    await flushMicrotasks();
    await send;
    await flushMicrotasks();

    expect(sendNewMessage).toHaveBeenCalledTimes(1);
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Couldn't send message",
        description: expect.stringContaining("network down"),
        variant: "destructive",
      }),
    );
  });

  it("resolves the follow-up once dispatched rather than when its answer ends", async () => {
    const view = renderHook(() => useCopilotPage());
    // The new turn's promise only settles when its whole stream ends; the
    // composer must get the box back long before that.
    sendNewMessage.mockReturnValue(new Promise(() => undefined));

    const send = view.result.current.onSend("follow-up");
    await flushMicrotasks();
    setStream({ status: "ready" });
    view.rerender();
    await flushMicrotasks();

    await send;
    expect(sendNewMessage).toHaveBeenCalledTimes(1);
  });
});
