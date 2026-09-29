import { TEST_BACKEND_BASE_URL } from "@/app/(platform)/copilot/__tests__/sse-helpers";
import { useCopilotStreamStore } from "@/app/(platform)/copilot/copilotStreamStore";
import {
  getKickoffStatus,
  markKickoffDone,
  markKickoffPending,
} from "@/app/(platform)/copilot/expertKickoff";
import {
  getGetV2GetSessionMockHandler200,
  getGetV2GetSessionResponseMock200,
  getGetV2ListSessionsMockHandler200,
  getPostV2CreateSessionMockHandler200,
  getPostV2CreateSessionResponseMock200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse, ws } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { onboardingCard, onboardingTurn } from "./onboardingFixtures";
import { ExpertChatDrawer } from "../ExpertChatDrawer";

const USER_ID = "user-1";
const EXPERT_ID = "3f8b0f7e-9f30-4a3b-a6a1-000000000001";
const SESSION_ID = "session-zara";
const FRESH_SESSION_ID = "session-zara-fresh";
const RETRY_SESSION_ID = "session-zara-retry";
const ERROR_SESSION_ID = "session-zara-error";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => TEST_BACKEND_BASE_URL,
    },
  };
});

vi.mock("@/app/(platform)/copilot/helpers", async (importActual) => {
  const actual =
    await importActual<typeof import("@/app/(platform)/copilot/helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

const backendSocket = ws.link("ws://localhost:8001/ws");

const ZARA = {
  expertId: EXPERT_ID,
  name: "Zara",
  role: "GTM Strategist",
  avatarUrl: null,
};

function freshThreadHandlers(
  createBodies: unknown[],
  streamBodies: string[],
  sessionId = FRESH_SESSION_ID,
) {
  return [
    getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
    getPostV2CreateSessionMockHandler200(async (info) => {
      createBodies.push(await info.request.clone().json());
      return getPostV2CreateSessionResponseMock200({ id: sessionId });
    }),
    getGetV2GetSessionMockHandler200(
      getGetV2GetSessionResponseMock200({
        id: sessionId,
        expert_id: EXPERT_ID,
        messages: [],
        active_stream: null,
      }),
    ),
    http.post(
      `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${sessionId}/stream`,
      async ({ request }) => {
        streamBodies.push(await request.clone().text());
        return streamSseResponse(assistantTextChunks("Hi, I'm Zara."), {
          abortSignal: request.signal,
        });
      },
    ),
  ];
}

beforeEach(() => {
  window.localStorage.clear();
  useCopilotStreamStore.getState().resetAll();
  server.use(backendSocket.addEventListener("connection", () => {}));
  useAuthStore.setState({
    user: { id: USER_ID, email: "zara-owner@example.com", user_metadata: {} },
    isUserLoading: false,
    hasLoadedUser: true,
  });
});

afterEach(() => {
  useAuthStore.setState({ user: null, hasLoadedUser: false });
});

describe("ExpertChatDrawer", () => {
  test("keeps chat disabled while it checks for an existing thread", async () => {
    const sessionsRequest = deferred();
    const createBodies: unknown[] = [];
    server.use(
      getGetV2ListSessionsMockHandler200(async () => {
        await sessionsRequest.promise;
        return { sessions: [], total: 0 };
      }),
      ...freshThreadHandlers(createBodies, []),
    );

    render(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
      />,
    );

    expect(
      (await screen.findByPlaceholderText(
        "Message Zara…",
      )) as HTMLTextAreaElement,
    ).toHaveProperty("disabled", true);
    expect(createBodies).toEqual([]);

    sessionsRequest.resolve();
    await waitFor(() => expect(createBodies.length).toBe(1));
    expect(await screen.findByText("Hi, I'm Zara.")).toBeDefined();
  });

  test("waits for a fresh session list before starting onboarding", async () => {
    const createBodies: unknown[] = [];
    server.use(...freshThreadHandlers(createBodies, []));
    const { rerender } = render(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
        threadKey={0}
      />,
    );

    await waitFor(() => expect(createBodies.length).toBe(1));
    expect(await screen.findByText("Hi, I'm Zara.")).toBeDefined();

    rerender(
      <ExpertChatDrawer
        target={null}
        onClose={() => {}}
        resumeLatest={false}
        threadKey={1}
      />,
    );
    await waitFor(() => expect(screen.queryByText("Hi, I'm Zara.")).toBeNull());
    window.localStorage.clear();

    const sessionsRequest = deferred();
    server.use(
      getGetV2ListSessionsMockHandler200(async () => {
        await sessionsRequest.promise;
        return {
          sessions: [
            {
              id: SESSION_ID,
              created_at: "2026-09-27T18:00:00Z",
              updated_at: "2026-09-27T18:01:00Z",
              is_processing: false,
              expert_id: EXPERT_ID,
            },
          ],
          total: 1,
        };
      }),
    );
    rerender(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
        threadKey={2}
      />,
    );

    expect(
      (await screen.findByPlaceholderText(
        "Message Zara…",
      )) as HTMLTextAreaElement,
    ).toHaveProperty("disabled", true);
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(createBodies.length).toBe(1);

    sessionsRequest.resolve();
    await waitFor(() =>
      expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("done"),
    );
    expect(createBodies.length).toBe(1);
  });

  test("retries a failed kickoff before sending the user's message", async () => {
    const user = userEvent.setup();
    const createBodies: unknown[] = [];
    const streamBodies: string[] = [];
    let createAttempts = 0;
    server.use(
      http.post("/api/proxy/api/chat/sessions", async ({ request }) => {
        createBodies.push(await request.clone().json());
        createAttempts += 1;
        if (createAttempts === 1) {
          return HttpResponse.json(
            { detail: "Could not create session" },
            { status: 500 },
          );
        }
        return HttpResponse.json(
          getPostV2CreateSessionResponseMock200({ id: RETRY_SESSION_ID }),
        );
      }),
      ...freshThreadHandlers([], streamBodies, RETRY_SESSION_ID),
    );

    render(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
      />,
    );

    await waitFor(() => expect(createAttempts).toBe(1));
    const input = await screen.findByPlaceholderText("Message Zara…");
    await waitFor(() =>
      expect(input as HTMLTextAreaElement).toHaveProperty("disabled", false),
    );
    await user.type(input, "Plan my launch{Enter}");

    await waitFor(() => expect(createAttempts).toBe(2));
    await waitFor(() => expect(streamBodies.length).toBe(2));
    expect(JSON.parse(streamBodies[0]).expert_kickoff).toBe(true);
    expect(JSON.parse(streamBodies[1])).toMatchObject({
      expert_kickoff: false,
      message: "Plan my launch",
    });
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("done");
    expect(createBodies).toEqual([
      { expert_id: EXPERT_ID, expert_kickoff: true },
      { expert_id: EXPERT_ID, expert_kickoff: true },
    ]);
  });

  test("releases the kickoff after its stream fails", async () => {
    let streamStarted = false;
    server.use(
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${ERROR_SESSION_ID}/stream`,
        () => {
          streamStarted = true;
          return HttpResponse.json(
            { detail: "Could not start stream" },
            { status: 500 },
          );
        },
      ),
      ...freshThreadHandlers([], [], ERROR_SESSION_ID),
    );

    render(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
      />,
    );

    await waitFor(() => expect(streamStarted).toBe(true));
    await waitFor(() =>
      expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("idle"),
    );
  });

  test("returns a prompt queued behind the kickoff when its stream fails", async () => {
    const user = userEvent.setup();
    const streamBodies: string[] = [];
    let createAttempts = 0;
    server.use(
      http.post("/api/proxy/api/chat/sessions", () => {
        createAttempts += 1;
        if (createAttempts === 1) {
          return HttpResponse.json(
            { detail: "Could not create session" },
            { status: 500 },
          );
        }
        return HttpResponse.json(
          getPostV2CreateSessionResponseMock200({ id: ERROR_SESSION_ID }),
        );
      }),
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${ERROR_SESSION_ID}/stream`,
        async ({ request }) => {
          streamBodies.push(await request.clone().text());
          return HttpResponse.json(
            { detail: "Could not start stream" },
            { status: 500 },
          );
        },
      ),
      ...freshThreadHandlers([], [], ERROR_SESSION_ID),
    );

    render(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
      />,
    );

    await waitFor(() => expect(createAttempts).toBe(1));
    const input = (await screen.findByPlaceholderText(
      "Message Zara…",
    )) as HTMLTextAreaElement;
    await waitFor(() => expect(input).toHaveProperty("disabled", false));
    await user.type(input, "Plan my launch{Enter}");

    await waitFor(() => expect(streamBodies.length).toBe(1));
    expect(JSON.parse(streamBodies[0]).expert_kickoff).toBe(true);
    await waitFor(() => expect(input.value).toBe("Plan my launch"));
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("idle");
  });

  test("opens a plain thread when the kickoff check fails", async () => {
    const user = userEvent.setup();
    const createBodies: unknown[] = [];
    const streamBodies: string[] = [];
    server.use(
      http.get("/api/proxy/api/chat/sessions", () =>
        HttpResponse.json({ detail: "Unavailable" }, { status: 503 }),
      ),
      ...freshThreadHandlers(createBodies, streamBodies),
    );

    render(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
      />,
    );

    expect(await screen.findByText("What can I do for you?")).toBeDefined();
    const input = await screen.findByPlaceholderText("Message Zara…");
    await waitFor(() =>
      expect(input as HTMLTextAreaElement).toHaveProperty("disabled", false),
    );
    await user.type(input, "Plan my launch{Enter}");

    await waitFor(() =>
      expect(createBodies).toEqual([{ expert_id: EXPERT_ID }]),
    );
    await waitFor(() => expect(streamBodies.length).toBe(1));
    expect(JSON.parse(streamBodies[0])).toMatchObject({
      expert_kickoff: false,
      message: "Plan my launch",
    });
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("idle");
  });

  test("kicks off onboarding the first time an expert's thread opens", async () => {
    const createBodies: unknown[] = [];
    const streamBodies: string[] = [];
    server.use(...freshThreadHandlers(createBodies, streamBodies));

    render(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
      />,
    );

    await waitFor(() =>
      expect(createBodies).toEqual([
        { expert_id: EXPERT_ID, expert_kickoff: true },
      ]),
    );
    await waitFor(() => expect(streamBodies.length).toBe(1));
    const body = JSON.parse(streamBodies[0]);
    expect(body.expert_kickoff).toBe(true);
    expect(body.message).toContain("expert_onboarding");
    expect(await screen.findByText("Hi, I'm Zara.")).toBeDefined();
    expect(screen.queryByText(/You were just hired/)).toBeNull();
    expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("done");
  });

  test("opens a plain thread once the expert has been onboarded", async () => {
    markKickoffDone(USER_ID, EXPERT_ID, markKickoffPending(USER_ID, EXPERT_ID));
    const createBodies: unknown[] = [];
    server.use(...freshThreadHandlers(createBodies, []));

    render(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
      />,
    );

    expect(await screen.findByText("What can I do for you?")).toBeDefined();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(createBodies).toEqual([]);
  });

  test("treats an existing thread as already onboarded", async () => {
    const createBodies: unknown[] = [];
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          {
            id: SESSION_ID,
            created_at: "2026-09-27T18:00:00Z",
            updated_at: "2026-09-27T18:01:00Z",
            is_processing: false,
            expert_id: EXPERT_ID,
          },
        ],
        total: 1,
      }),
      ...freshThreadHandlers(createBodies, []),
    );

    render(
      <ExpertChatDrawer
        target={ZARA}
        onClose={() => {}}
        resumeLatest={false}
      />,
    );

    expect(await screen.findByText("What can I do for you?")).toBeDefined();
    await waitFor(() =>
      expect(getKickoffStatus(USER_ID, EXPERT_ID)).toBe("done"),
    );
    expect(createBodies).toEqual([]);
  });

  test("keeps answers if creating the new chat fails", async () => {
    markKickoffDone(USER_ID, EXPERT_ID, markKickoffPending(USER_ID, EXPERT_ID));
    const createSession = vi.fn(() =>
      HttpResponse.json({ detail: "Unavailable" }, { status: 503 }),
    );
    server.use(
      http.get("/api/proxy/api/experts/:expertId/onboarding", () =>
        HttpResponse.json(onboardingCard()),
      ),
      http.post("/api/proxy/api/chat/sessions", createSession),
    );
    render(
      <ExpertChatDrawer
        target={ZARA}
        resumeLatest={false}
        onClose={() => {}}
      />,
    );
    fireEvent.click(await screen.findByRole("radio", { name: "Pricing" }));
    fireEvent.click(screen.getByRole("button", { name: "Send answers" }));
    await waitFor(() => expect(createSession).toHaveBeenCalled());
    await waitFor(() =>
      expect(
        (
          screen.getByRole("button", {
            name: "Send answers",
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(false),
    );
    expect(
      screen
        .getByRole("radio", { name: "Pricing" })
        .getAttribute("aria-checked"),
    ).toBe("true");
  });

  test("keeps settled setup out of new chats", async () => {
    markKickoffDone(USER_ID, EXPERT_ID, markKickoffPending(USER_ID, EXPERT_ID));
    const request = vi.fn(() => HttpResponse.json(null));
    server.use(
      http.get("/api/proxy/api/experts/:expertId/onboarding", request),
    );
    render(
      <ExpertChatDrawer
        target={ZARA}
        resumeLatest={false}
        onClose={() => {}}
      />,
    );
    await waitFor(() => expect(request).toHaveBeenCalled());
    expect(screen.getByText("What can I do for you?")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Skip" })).toBeNull();
  });

  test("does not request setup when the new chat has a prompt", async () => {
    markKickoffDone(USER_ID, EXPERT_ID, markKickoffPending(USER_ID, EXPERT_ID));
    const request = vi.fn(() => HttpResponse.json(onboardingCard()));
    server.use(
      http.get("/api/proxy/api/experts/:expertId/onboarding", request),
    );
    render(
      <ExpertChatDrawer
        target={ZARA}
        resumeLatest={false}
        seedPrompt="Research this company"
        onClose={() => {}}
      />,
    );
    expect(await screen.findByText("What can I do for you?")).toBeDefined();
    expect(request).not.toHaveBeenCalled();
  });

  test("shows unanswered setup questions in a new chat", async () => {
    markKickoffDone(USER_ID, EXPERT_ID, markKickoffPending(USER_ID, EXPERT_ID));
    server.use(
      http.get("/api/proxy/api/experts/:expertId/onboarding", () =>
        HttpResponse.json(onboardingCard()),
      ),
    );
    render(
      <ExpertChatDrawer
        target={ZARA}
        resumeLatest={false}
        onClose={() => {}}
      />,
    );
    expect(
      await screen.findByText("Which outcome should I start with?"),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Skip" })).toBeDefined();
  });

  test("a hire's onboarding card is a live form, not a settled row", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          {
            id: SESSION_ID,
            created_at: "2026-09-27T18:00:00Z",
            updated_at: "2026-09-27T18:01:00Z",
            is_processing: false,
            expert_id: EXPERT_ID,
          },
        ],
        total: 1,
      }),
      getGetV2GetSessionMockHandler200({
        id: SESSION_ID,
        created_at: "2026-09-27T18:00:00Z",
        updated_at: "2026-09-27T18:01:00Z",
        user_id: "user-1",
        expert_id: EXPERT_ID,
        messages: onboardingTurn(),
      }),
    );

    render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

    expect(
      await screen.findByText("Which outcome should I start with?"),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Skip" })).toBeDefined();
    expect(screen.queryByText(/Setup questions from/)).toBeNull();
  });
});
