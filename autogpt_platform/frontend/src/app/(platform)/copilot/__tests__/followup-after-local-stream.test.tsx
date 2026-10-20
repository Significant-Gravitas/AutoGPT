import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// A follow-up sent while this tab is still drawing the previous answer, but
// after the backend's turn has ended: `/messages/pending` answers 409, and
// before this guard the fallback started a second `useChat` request on top
// of the live stream, which froze the answer mid-sentence (SECRT-2772).
// The session-id mock is static on purpose: the chat host is keyed by
// session id, so a chat switch unmounts this hook rather than re-rendering
// it with a new id.

const streamState = vi.hoisted(() => ({
  status: "ready" as "ready" | "submitted" | "streaming" | "error",
  isFinishProbing: false,
  isReconnecting: false,
  isUserStopping: false,
}));
const sessionState = vi.hoisted(() => ({ sessionId: "session-1" }));

const sendNewMessage = vi.hoisted(() => vi.fn());
const queueFollowUpMessage = vi.hoisted(() => vi.fn());
const queueMessage = vi.hoisted(() => vi.fn());
const toast = vi.hoisted(() => vi.fn());

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

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast,
  useToast: () => ({ toast, dismiss: vi.fn() }),
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
    isSyncing: false,
    isUserStoppingRef: { current: false },
    isUserStopping: streamState.isUserStopping,
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
      setPendingFileParts: vi.fn(),
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
vi.mock("../useWorkflowImportAutoSubmit", () => ({
  useWorkflowImportAutoSubmit: () => undefined,
}));
vi.mock("../useExpertKickoff", () => ({
  useExpertKickoff: () => ({ isKickoffStarting: false }),
}));
vi.mock("../helpers/queueFollowUpMessage", () => ({ queueFollowUpMessage }));

import { useCopilotUIStore } from "../store";
import { useCopilotPage } from "../useCopilotPage";
import * as copilotHelpers from "../helpers";

function noActiveTurn() {
  const error = new Error("Session has no active turn to queue against.");
  error.name = "QueueFollowUpNotActiveError";
  return error;
}

async function flush() {
  await act(async () => {
    await Promise.resolve();
  });
}

beforeEach(() => {
  streamState.status = "streaming";
  streamState.isFinishProbing = false;
  streamState.isReconnecting = false;
  streamState.isUserStopping = false;
  sessionState.sessionId = "session-1";
  sendNewMessage.mockResolvedValue(undefined);
  queueFollowUpMessage.mockRejectedValue(noActiveTurn());
  useCopilotUIStore.getState().setInitialPrompt(null);
});

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
  window.sessionStorage.clear();
});

describe("useCopilotPage — follow-up refused by the backend mid-stream", () => {
  it.each(["ready", "error"] as const)(
    "starts a new turn directly after the stream settles with status %s",
    async (status) => {
      const { result, rerender } = renderHook(() => useCopilotPage());

      streamState.status = status;
      rerender();
      await result.current.onSend("Retry the failed request");

      expect(queueFollowUpMessage).not.toHaveBeenCalled();
      expect(sendNewMessage).toHaveBeenCalledWith(
        "Retry the failed request",
        undefined,
        undefined,
        undefined,
      );
    },
  );

  it.each(["ready", "error"] as const)(
    "starts a new turn after a real pending-message 409 and local status %s",
    async (status) => {
      const actualQueue = await vi.importActual<
        typeof import("../helpers/queueFollowUpMessage")
      >("../helpers/queueFollowUpMessage");
      queueFollowUpMessage.mockImplementation(actualQueue.queueFollowUpMessage);
      vi.spyOn(copilotHelpers, "getCopilotAuthHeaders").mockResolvedValue({});
      const fetchMock = vi.spyOn(global, "fetch").mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            detail:
              "Session has no active turn. Start a new turn with POST /stream.",
          }),
          { status: 409 },
        ),
      );
      const { result, rerender } = renderHook(() => useCopilotPage());

      const send = result.current.onSend("Retry the failed request");
      await flush();
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining(
          "/api/chat/sessions/session-1/messages/pending",
        ),
        expect.objectContaining({ method: "POST" }),
      );
      expect(sendNewMessage).not.toHaveBeenCalled();

      streamState.status = status;
      rerender();
      await send;

      expect(sendNewMessage).toHaveBeenCalledWith(
        "Retry the failed request",
        undefined,
        undefined,
        undefined,
      );
      expect(queueMessage).not.toHaveBeenCalled();
      expect(toast).not.toHaveBeenCalled();
    },
  );

  it("holds the follow-up until this tab's stream has settled", async () => {
    const { result, rerender } = renderHook(() => useCopilotPage());

    let settled = false;
    const send = result.current.onSend("After the essay, say PINEAPPLE");
    void send.then(() => (settled = true));
    await flush();

    expect(queueFollowUpMessage).toHaveBeenCalledWith(
      "session-1",
      "After the essay, say PINEAPPLE",
    );
    expect(sendNewMessage).not.toHaveBeenCalled();
    expect(settled).toBe(false);

    streamState.status = "ready";
    rerender();
    await send;

    expect(sendNewMessage).toHaveBeenCalledTimes(1);
    expect(sendNewMessage).toHaveBeenCalledWith(
      "After the essay, say PINEAPPLE",
      undefined,
      undefined,
      undefined,
    );
    expect(queueMessage).not.toHaveBeenCalled();
  });

  it("waits out the post-finish probe and a reconnect as well", async () => {
    const { result, rerender } = renderHook(() => useCopilotPage());
    const send = result.current.onSend("and then?");
    await flush();

    streamState.status = "ready";
    streamState.isFinishProbing = true;
    rerender();
    await flush();
    expect(sendNewMessage).not.toHaveBeenCalled();

    streamState.isFinishProbing = false;
    streamState.isReconnecting = true;
    rerender();
    await flush();
    expect(sendNewMessage).not.toHaveBeenCalled();

    streamState.isReconnecting = false;
    rerender();
    await send;
    expect(sendNewMessage).toHaveBeenCalledTimes(1);
  });

  it("resolves once the follow-up is dispatched, not when its answer ends", async () => {
    sendNewMessage.mockReturnValue(new Promise(() => {}));
    const { result, rerender } = renderHook(() => useCopilotPage());
    const send = result.current.onSend("and then?");
    await flush();

    streamState.status = "ready";
    rerender();
    await expect(send).resolves.toBeUndefined();
    expect(sendNewMessage).toHaveBeenCalledTimes(1);
  });

  it("queues a second held follow-up behind the first instead of starting two turns", async () => {
    const { result, rerender } = renderHook(() => useCopilotPage());
    const first = result.current.onSend("first");
    const second = result.current.onSend("second");
    await flush();
    expect(queueFollowUpMessage).toHaveBeenCalledTimes(2);

    // Once the first has dispatched, the backend has a turn to queue against.
    queueFollowUpMessage.mockResolvedValue({
      buffer_length: 1,
      max_buffer_length: 10,
      turn_in_flight: true,
    });
    streamState.status = "ready";
    rerender();
    await Promise.all([first, second]);

    expect(sendNewMessage).toHaveBeenCalledTimes(1);
    expect(sendNewMessage).toHaveBeenCalledWith(
      "first",
      undefined,
      undefined,
      undefined,
    );
    expect(queueMessage).toHaveBeenCalledWith("second");
    // Both held copies are released: one went out, one is now the
    // backend's chip.
    await waitFor(() => expect(result.current.queuedMessages).toEqual([]));
    expect(window.sessionStorage.length).toBe(0);
  });

  it("still queues normally when the backend accepts the follow-up", async () => {
    queueFollowUpMessage.mockResolvedValue({
      buffer_length: 1,
      max_buffer_length: 10,
      turn_in_flight: true,
    });
    const { result } = renderHook(() => useCopilotPage());
    await result.current.onSend("and then?");

    expect(queueMessage).toHaveBeenCalledWith("and then?");
    expect(sendNewMessage).not.toHaveBeenCalled();
  });

  it("shows the held follow-up as a queued chip until it goes out", async () => {
    const { result, rerender } = renderHook(() => useCopilotPage());
    const send = result.current.onSend("and then?");
    await flush();
    expect(result.current.queuedMessages).toEqual(["and then?"]);
    expect(
      window.sessionStorage.getItem("copilot-held-follow-ups:session-1"),
    ).toBe(JSON.stringify(["and then?"]));

    streamState.status = "ready";
    rerender();
    await send;
    await waitFor(() => expect(result.current.queuedMessages).toEqual([]));
    expect(window.sessionStorage.length).toBe(0);
  });

  it("gives up the send, but not the copy, when the chat host unmounts mid-hold", async () => {
    const { result, unmount } = renderHook(() => useCopilotPage());
    const send = result.current.onSend("and then?");
    await flush();

    unmount();
    await expect(send).resolves.toBeUndefined();
    expect(sendNewMessage).not.toHaveBeenCalled();
    expect(
      window.sessionStorage.getItem("copilot-held-follow-ups:session-1"),
    ).toBe(JSON.stringify(["and then?"]));
  });

  it("lets a queue failure reach the composer once, without a toast of its own", async () => {
    queueFollowUpMessage.mockRejectedValue(new Error("Expected 200; got 500"));
    const { result } = renderHook(() => useCopilotPage());

    await expect(result.current.onSend("and then?")).rejects.toThrow(
      "Expected 200; got 500",
    );
    expect(toast).not.toHaveBeenCalled();
    expect(sendNewMessage).not.toHaveBeenCalled();
    expect(result.current.queuedMessages).toEqual([]);
  });

  it("puts a follow-up left behind by a reload back in the composer", async () => {
    window.sessionStorage.setItem(
      "copilot-held-follow-ups:session-1",
      JSON.stringify(["and then?"]),
    );
    renderHook(() => useCopilotPage());

    await waitFor(() =>
      expect(useCopilotUIStore.getState().initialPrompt).toBe("and then?"),
    );
    expect(window.sessionStorage.length).toBe(0);
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Follow-up not sent" }),
    );
    expect(sendNewMessage).not.toHaveBeenCalled();
  });
});
