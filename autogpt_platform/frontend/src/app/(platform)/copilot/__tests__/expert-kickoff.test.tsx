import { server } from "@/mocks/mock-server";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { UIMessage } from "ai";
import { http, HttpResponse } from "msw";
import { parseAsString, useQueryState } from "nuqs";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import {
  buildKickoffMessage,
  getKickoffAttemptToken,
  getKickoffStatus,
  isKickoffMessage,
  kickoffStorageKey,
  markKickoffDone,
  markKickoffPending,
} from "../expertKickoff";
import { useCopilotUIStore } from "../store";
import { useChatSession } from "../useChatSession";
import { useExpertKickoff } from "../useExpertKickoff";
import { useSendMessage } from "../useSendMessage";

const flagState = vi.hoisted(() => ({
  values: { "hire-experts": true } as Record<string, boolean>,
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flagState.values[flag] ?? false,
  };
});

const sendSpy = vi.hoisted(() => vi.fn());

const USER_ID = "user-1";
const EXPERT_ID = "3f8b0f7e-9f30-4a3b-a6a1-000000000001";
const KICKOFF_SESSION_ID = "4f8b0f7e-9f30-4a3b-a6a1-000000000001";
const EXISTING_SESSION_ID = "4f8b0f7e-9f30-4a3b-a6a1-000000000002";
const ORPHAN_SESSION_ID = "4f8b0f7e-9f30-4a3b-a6a1-000000000003";

function latestKickoffAttemptToken(messages: UIMessage[]) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const attemptToken = getKickoffAttemptToken(messages[index]);
    if (attemptToken) return attemptToken;
  }
  return null;
}

afterEach(() => {
  server.resetHandlers();
  sendSpy.mockReset();
  flagState.values = { "hire-experts": true };
  window.localStorage.clear();
  useCopilotUIStore.setState({ adoptedExpertThreads: new Set<string>() });
  useCopilotStreamStore.getState().resetAll();
});

function KickoffHarness() {
  const [expertId] = useQueryState("expertId", parseAsString);
  const [kickoff] = useQueryState("kickoff", parseAsString);
  const [clientMessages, setClientMessages] = useState<UIMessage[]>([]);
  const [settledCount, setSettledCount] = useState(0);
  const {
    sessionId,
    setSessionId,
    sessionExpertId,
    hydratedMessages,
    createSession,
    refetchSession,
  } = useChatSession({ expertId });
  const isUserStoppingRef = { current: false };

  async function sendMessage(input: {
    text?: string;
    metadata?: UIMessage["metadata"];
  }) {
    sendSpy(input);
    setClientMessages((messages) => [
      ...messages,
      {
        id: `client-${messages.length + 1}`,
        role: "user",
        parts: [{ type: "text", text: input.text ?? "" }],
        metadata: input.metadata,
      },
    ]);
  }

  const { onSend } = useSendMessage({
    sessionId,
    sendMessage: sendMessage as never,
    createSession,
    isUserStoppingRef,
  });

  const { isKickoffStarting } = useExpertKickoff({
    userId: USER_ID,
    expertId,
    kickoff: kickoff === "1",
    sessionId,
    sessionExpertId,
    hasPersistedExpertHistory:
      sessionId && hydratedMessages !== undefined
        ? hydratedMessages.some((message) => !isKickoffMessage(message))
        : null,
    kickoffAttemptToken: latestKickoffAttemptToken(clientMessages),
    isClientThreadEmpty: clientMessages.every(isKickoffMessage),
    onAdoptSession: setSessionId,
    async onKickoff(id, attemptToken) {
      const message = buildKickoffMessage(id, attemptToken);
      await onSend(message.text, undefined, undefined, message.metadata);
    },
    onSettled() {
      setSettledCount((count) => count + 1);
    },
  });

  return (
    <div>
      <div data-testid="session-id">{sessionId ?? "none"}</div>
      <div data-testid="kickoff-param">{kickoff ?? "none"}</div>
      <div data-testid="settled-count">{settledCount}</div>
      <div data-testid="kickoff-starting">{String(isKickoffStarting)}</div>
      <div data-testid="client-message-count">{clientMessages.length}</div>
      <button type="button" onClick={() => void refetchSession()}>
        Refresh session
      </button>
    </div>
  );
}

function renderKickoff(searchParams: string) {
  const Wrapper = withNuqsTestingAdapter({ searchParams, hasMemory: true });
  return render(
    <CredentialsProvidersContext.Provider value={{}}>
      <Wrapper>
        <KickoffHarness />
      </Wrapper>
    </CredentialsProvidersContext.Provider>,
  );
}

function stubTransports() {
  return http.get("*/api/chat/transports", () =>
    HttpResponse.json({
      transports: [
        {
          auth_provider: "platform",
          credential_id: null,
          label: "AutoGPT Platform",
          available: true,
          default: true,
        },
      ],
    }),
  );
}

function makeSessionPayload(id: string, messages: unknown[] = []) {
  return {
    id,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-02T00:00:00Z",
    user_id: USER_ID,
    expert_id: EXPERT_ID,
    messages,
  };
}

function persistedKickoffMessage() {
  const kickoff = buildKickoffMessage(EXPERT_ID);
  return {
    id: "5f8b0f7e-9f30-4a3b-a6a1-000000000001",
    role: "user",
    content: kickoff.text,
    metadata: {
      hidden: true,
      kind: "expert_kickoff",
      expert_id: EXPERT_ID,
    },
    created_at: "2026-01-01T00:00:00Z",
  };
}

function persistedAssistantMessage() {
  return {
    id: "5f8b0f7e-9f30-4a3b-a6a1-000000000002",
    role: "assistant",
    content: "I am ready to help.",
    created_at: "2026-01-01T00:00:01Z",
  };
}

describe("useExpertKickoff", () => {
  it("creates atomically and waits for server persistence before consuming kickoff", async () => {
    let createBody: unknown = null;
    let createCount = 0;
    let persistedMessages: unknown[] = [];
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({ sessions: [], total: 0 }),
      ),
      http.post("*/api/chat/sessions", async ({ request }) => {
        createCount += 1;
        createBody = await request.json();
        return HttpResponse.json({
          id: KICKOFF_SESSION_ID,
          created_at: "2026-01-01T00:00:00Z",
          user_id: USER_ID,
          expert_id: EXPERT_ID,
        });
      }),
      http.get(`*/api/chat/sessions/${KICKOFF_SESSION_ID}`, () =>
        HttpResponse.json(
          makeSessionPayload(KICKOFF_SESSION_ID, persistedMessages),
        ),
      ),
    );

    renderKickoff(`?expertId=${EXPERT_ID}&kickoff=1`);

    await waitFor(() => expect(createCount).toBe(1));
    expect(createBody).toEqual({
      expert_id: EXPERT_ID,
      expert_kickoff: true,
      llm_auth_provider: "platform",
    });

    await waitFor(() => expect(sendSpy).toHaveBeenCalledTimes(1));
    expect(sendSpy).toHaveBeenCalledWith({
      text: buildKickoffMessage(EXPERT_ID).text,
      files: undefined,
      metadata: {
        kind: "expert_kickoff",
        expertId: EXPERT_ID,
        attemptToken: expect.any(String),
      },
    });
    expect(screen.getByTestId("client-message-count").textContent).toBe("1");
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("pending");
    expect(screen.getByTestId("kickoff-param").textContent).toBe("1");
    expect(screen.getByTestId("kickoff-starting").textContent).toBe("true");

    persistedMessages = [
      persistedKickoffMessage(),
      persistedAssistantMessage(),
    ];
    fireEvent.click(screen.getByRole("button", { name: "Refresh session" }));

    await waitFor(() =>
      expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("done"),
    );
    await waitFor(() =>
      expect(screen.getByTestId("settled-count").textContent).toBe("1"),
    );
    expect(screen.getByTestId("kickoff-starting").textContent).toBe("false");
    expect(sendSpy).toHaveBeenCalledTimes(1);
    expect(createCount).toBe(1);
  });

  it("does nothing on a second visit after persisted completion", async () => {
    const attemptToken = markKickoffPending(USER_ID, EXPERT_ID);
    markKickoffDone(USER_ID, EXPERT_ID, attemptToken);
    let listCount = 0;
    let createCount = 0;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () => {
        listCount += 1;
        return HttpResponse.json({ sessions: [], total: 0 });
      }),
      http.post("*/api/chat/sessions", () => {
        createCount += 1;
        return HttpResponse.json({ id: "should-not-happen" });
      }),
    );

    renderKickoff(`?expertId=${EXPERT_ID}&kickoff=1`);

    await waitFor(() => expect(listCount).toBeGreaterThan(0));
    await waitFor(() =>
      expect(screen.getByTestId("settled-count").textContent).toBe("1"),
    );
    expect(sendSpy).not.toHaveBeenCalled();
    expect(createCount).toBe(0);
    expect(screen.getByTestId("session-id").textContent).toBe("none");
  });

  it("stands down while another tab holds a fresh pending latch", async () => {
    markKickoffPending(USER_ID, EXPERT_ID);
    let listCount = 0;
    let createCount = 0;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () => {
        listCount += 1;
        return HttpResponse.json({ sessions: [], total: 0 });
      }),
      http.post("*/api/chat/sessions", () => {
        createCount += 1;
        return HttpResponse.json({ id: "should-not-happen" });
      }),
    );

    renderKickoff(`?expertId=${EXPERT_ID}&kickoff=1`);

    await waitFor(() => expect(listCount).toBeGreaterThan(0));
    expect(sendSpy).not.toHaveBeenCalled();
    expect(createCount).toBe(0);
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("pending");
  });

  it("releases pending state when atomic session creation fails", async () => {
    let createCount = 0;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({ sessions: [], total: 0 }),
      ),
      http.post("*/api/chat/sessions", () => {
        createCount += 1;
        return new HttpResponse(null, { status: 500 });
      }),
    );

    renderKickoff(`?expertId=${EXPERT_ID}&kickoff=1`);

    await waitFor(() => expect(createCount).toBe(1));
    await waitFor(() =>
      expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("idle"),
    );
    expect(sendSpy).not.toHaveBeenCalled();
    expect(
      window.localStorage.getItem(kickoffStorageKey(USER_ID, EXPERT_ID)),
    ).toBeNull();
    expect(screen.getByTestId("kickoff-starting").textContent).toBe("false");
    expect(screen.getByTestId("settled-count").textContent).toBe("1");
  });

  it("adopts a thread with history and retires kickoff without sending", async () => {
    let createCount = 0;
    let requestedExpertId: string | null = null;
    let requestedPinnedFirst: string | null = null;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", ({ request }) => {
        const url = new URL(request.url);
        requestedExpertId = url.searchParams.get("expert_id");
        requestedPinnedFirst = url.searchParams.get("pinned_first");
        return HttpResponse.json({
          sessions: [
            {
              id: EXISTING_SESSION_ID,
              title: "Maria thread",
              expert_id: EXPERT_ID,
              is_processing: false,
              created_at: "2026-01-01T00:00:00Z",
              updated_at: "2026-01-02T00:00:00Z",
            },
          ],
          total: 1,
        });
      }),
      http.get(`*/api/chat/sessions/${EXISTING_SESSION_ID}`, () =>
        HttpResponse.json(
          makeSessionPayload(EXISTING_SESSION_ID, [
            {
              id: "5f8b0f7e-9f30-4a3b-a6a1-000000000002",
              role: "user",
              content: "Plan my week",
              created_at: "2026-01-01T00:00:00Z",
            },
          ]),
        ),
      ),
      http.post("*/api/chat/sessions", () => {
        createCount += 1;
        return HttpResponse.json({ id: "should-not-happen" });
      }),
    );

    renderKickoff(`?expertId=${EXPERT_ID}&kickoff=1`);

    await waitFor(() =>
      expect(screen.getByTestId("session-id").textContent).toBe(
        EXISTING_SESSION_ID,
      ),
    );
    await waitFor(() =>
      expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("done"),
    );
    expect(screen.getByTestId("kickoff-starting").textContent).toBe("false");
    expect(requestedExpertId).toBe(EXPERT_ID);
    expect(requestedPinnedFirst).toBe("false");
    expect(sendSpy).not.toHaveBeenCalled();
    expect(createCount).toBe(0);
  });

  it("recovers an adopted empty expert session without duplicate creation", async () => {
    let createCount = 0;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({
          sessions: [
            {
              id: ORPHAN_SESSION_ID,
              title: "New chat",
              expert_id: EXPERT_ID,
              is_processing: false,
              created_at: "2026-01-01T00:00:00Z",
              updated_at: "2026-01-01T00:00:00Z",
            },
          ],
          total: 1,
        }),
      ),
      http.get(`*/api/chat/sessions/${ORPHAN_SESSION_ID}`, () =>
        HttpResponse.json(
          makeSessionPayload(ORPHAN_SESSION_ID, [persistedKickoffMessage()]),
        ),
      ),
      http.post("*/api/chat/sessions", () => {
        createCount += 1;
        return HttpResponse.json({ id: "should-not-happen" });
      }),
    );

    renderKickoff(`?expertId=${EXPERT_ID}&kickoff=1`);

    await waitFor(() =>
      expect(screen.getByTestId("session-id").textContent).toBe(
        ORPHAN_SESSION_ID,
      ),
    );
    await waitFor(() => expect(sendSpy).toHaveBeenCalledTimes(1));
    expect(sendSpy).toHaveBeenCalledWith({
      text: buildKickoffMessage(EXPERT_ID).text,
      files: undefined,
      metadata: {
        kind: "expert_kickoff",
        expertId: EXPERT_ID,
        attemptToken: expect.any(String),
      },
    });
    expect(createCount).toBe(0);
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("pending");
    expect(screen.getByTestId("kickoff-starting").textContent).toBe("true");
  });

  // `useCopilotPage` hands this hook `expertId: null` whenever the roster
  // refuses the id in the URL (fired, bogus, or unreadable) while `kickoff=1`
  // is still in the query string. Every path here has to stay inert on that
  // pair rather than reach storage or the network with a null id.
  it("stays inert when kickoff is armed without a resolvable expert", async () => {
    let listCount = 0;
    let createCount = 0;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () => {
        listCount += 1;
        return HttpResponse.json({ sessions: [], total: 0 });
      }),
      http.post("*/api/chat/sessions", () => {
        createCount += 1;
        return HttpResponse.json({ id: "should-not-happen" });
      }),
    );

    renderKickoff("?kickoff=1");

    await waitFor(() =>
      expect(screen.getByTestId("kickoff-starting").textContent).toBe("false"),
    );
    expect(sendSpy).not.toHaveBeenCalled();
    expect(listCount).toBe(0);
    expect(createCount).toBe(0);
    expect(screen.getByTestId("session-id").textContent).toBe("none");
    expect(screen.getByTestId("settled-count").textContent).toBe("0");
    expect(window.localStorage.length).toBe(0);
  });
});
