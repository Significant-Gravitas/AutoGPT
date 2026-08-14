import { server } from "@/mocks/mock-server";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import {
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { parseAsString, useQueryState } from "nuqs";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import {
  buildKickoffMessage,
  getKickoffStatus,
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

const EXPERT_ID = "expert-maria";

afterEach(() => {
  cleanup();
  server.resetHandlers();
  sendSpy.mockClear();
  flagState.values = { "hire-experts": true };
  window.localStorage.clear();
  useCopilotUIStore.setState({ adoptedExpertThreads: new Set<string>() });
  useCopilotStreamStore.getState().resetAll();
});

function KickoffHarness() {
  const [expertId] = useQueryState("expertId", parseAsString);
  const [kickoff] = useQueryState("kickoff", parseAsString);
  const {
    sessionId,
    setSessionId,
    sessionExpertId,
    hydratedMessages,
    createSession,
  } = useChatSession({ expertId });
  const isUserStoppingRef = { current: false };
  const { onSend } = useSendMessage({
    sessionId,
    sendMessage: sendSpy as never,
    createSession,
    isUserStoppingRef,
  });
  useExpertKickoff({
    expertId,
    kickoff: kickoff === "1",
    sessionId,
    sessionExpertId,
    isThreadEmpty:
      sessionId && hydratedMessages !== undefined
        ? hydratedMessages.length === 0
        : null,
    onAdoptSession: (id) => void setSessionId(id),
    onKickoff: (id) => onSend(buildKickoffMessage(id)),
  });
  return <div data-testid="session-id">{sessionId ?? "none"}</div>;
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
    user_id: "user-1",
    expert_id: EXPERT_ID,
    messages,
  };
}

describe("useExpertKickoff", () => {
  it("creates an expert session and sends the kickoff message exactly once", async () => {
    let createBody: unknown = null;
    let createCount = 0;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({ sessions: [], total: 0 }),
      ),
      http.post("*/api/chat/sessions", async ({ request }) => {
        createCount += 1;
        createBody = await request.json();
        return HttpResponse.json({
          id: "kickoff-session-1",
          created_at: "2026-01-01T00:00:00Z",
          user_id: "user-1",
          expert_id: EXPERT_ID,
        });
      }),
      http.get("*/api/chat/sessions/kickoff-session-1", () =>
        HttpResponse.json(makeSessionPayload("kickoff-session-1")),
      ),
    );

    renderKickoff(`?expertId=${EXPERT_ID}&kickoff=1`);

    await waitFor(() => expect(createCount).toBe(1));
    expect(createBody).toEqual({
      expert_id: EXPERT_ID,
      llm_auth_provider: "platform",
    });

    await waitFor(() => expect(sendSpy).toHaveBeenCalledTimes(1));
    const firstArg = sendSpy.mock.calls[0][0] as { text: string };
    expect(firstArg.text).toBe(buildKickoffMessage(EXPERT_ID));

    // The accepted send flips the latch to done, and nothing re-fires.
    await waitFor(() => expect(getKickoffStatus(EXPERT_ID)).toBe("done"));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sendSpy).toHaveBeenCalledTimes(1);
    expect(createCount).toBe(1);
  });

  it("no-ops on a second visit once the kickoff is done", async () => {
    markKickoffDone(EXPERT_ID);
    let createCount = 0;
    server.use(
      stubTransports(),
      // This list query is `useChatSession`'s adoption lookup, not the
      // kickoff's — returning empty keeps the user on the new-task screen.
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({ sessions: [], total: 0 }),
      ),
      http.post("*/api/chat/sessions", () => {
        createCount += 1;
        return HttpResponse.json({ id: "should-not-happen" });
      }),
    );

    renderKickoff(`?expertId=${EXPERT_ID}&kickoff=1`);

    await new Promise((resolve) => setTimeout(resolve, 80));
    expect(sendSpy).not.toHaveBeenCalled();
    expect(createCount).toBe(0);
    expect(screen.getByTestId("session-id").textContent).toBe("none");
  });

  it("stands down while another tab holds a fresh pending latch", async () => {
    markKickoffPending(EXPERT_ID);
    let createCount = 0;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({ sessions: [], total: 0 }),
      ),
      http.post("*/api/chat/sessions", () => {
        createCount += 1;
        return HttpResponse.json({ id: "should-not-happen" });
      }),
    );

    renderKickoff(`?expertId=${EXPERT_ID}&kickoff=1`);

    await new Promise((resolve) => setTimeout(resolve, 80));
    expect(sendSpy).not.toHaveBeenCalled();
    expect(createCount).toBe(0);
    // The other tab's pending latch is left untouched for it to resolve.
    expect(getKickoffStatus(EXPERT_ID)).toBe("pending");
  });

  it("releases the latch when session creation fails so a retry stays possible", async () => {
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
    // Failure surfaces via the standard error toast; the kickoff is NOT
    // consumed — the latch returns to idle so the next visit retries.
    await waitFor(() => expect(getKickoffStatus(EXPERT_ID)).toBe("idle"));
    expect(sendSpy).not.toHaveBeenCalled();
    expect(
      window.localStorage.getItem(kickoffStorageKey(EXPERT_ID)),
    ).toBeNull();
  });

  it("adopts a thread with history and retires the kickoff without sending", async () => {
    let createCount = 0;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({
          sessions: [
            {
              id: "existing-maria",
              title: "Maria thread",
              expert_id: EXPERT_ID,
              is_processing: false,
              created_at: "2026-01-01T00:00:00Z",
              updated_at: "2026-01-02T00:00:00Z",
            },
          ],
          total: 1,
        }),
      ),
      http.get("*/api/chat/sessions/existing-maria", () =>
        HttpResponse.json(
          makeSessionPayload("existing-maria", [
            {
              id: "m1",
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
        "existing-maria",
      ),
    );
    await waitFor(() => expect(getKickoffStatus(EXPERT_ID)).toBe("done"));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sendSpy).not.toHaveBeenCalled();
    expect(createCount).toBe(0);
  });

  it("resends the kickoff into an adopted EMPTY expert session (crashed tab recovery)", async () => {
    let createCount = 0;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({
          sessions: [
            {
              id: "orphan-kickoff",
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
      http.get("*/api/chat/sessions/orphan-kickoff", () =>
        HttpResponse.json(makeSessionPayload("orphan-kickoff")),
      ),
      http.post("*/api/chat/sessions", () => {
        createCount += 1;
        return HttpResponse.json({ id: "should-not-happen" });
      }),
    );

    renderKickoff(`?expertId=${EXPERT_ID}&kickoff=1`);

    await waitFor(() =>
      expect(screen.getByTestId("session-id").textContent).toBe(
        "orphan-kickoff",
      ),
    );
    // The empty session means the previous kickoff never landed — resend into
    // the existing thread rather than leaving a silent dead chat.
    await waitFor(() => expect(sendSpy).toHaveBeenCalledTimes(1));
    const firstArg = sendSpy.mock.calls[0][0] as { text: string };
    expect(firstArg.text).toBe(buildKickoffMessage(EXPERT_ID));
    expect(createCount).toBe(0);
    await waitFor(() => expect(getKickoffStatus(EXPERT_ID)).toBe("done"));
  });
});
