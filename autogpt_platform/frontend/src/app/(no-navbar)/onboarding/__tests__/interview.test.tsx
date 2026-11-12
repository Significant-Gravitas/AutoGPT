import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import { useCopilotStreamStore } from "@/app/(platform)/copilot/copilotStreamStore";
import {
  getFinalizeBrainDumpMockHandler200,
  getStartOnboardingInterviewMockHandler200,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import {
  getGetV2GetPendingMessagesMockHandler200,
  getGetV2GetSessionMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { FrontendFlag } from "@/app/api/__generated__/models/frontendFlag";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  createStreamHold,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import {
  act,
  cleanup,
  render,
  renderHook,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse, type JsonBodyType, ws } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import OnboardingPage from "../page";
import { readLocalProgress } from "../progress";
import { useInterviewStep } from "../steps/InterviewStep/useInterviewStep";
import { useOnboardingWizardStore } from "../store";
import { INTERVIEW_FLAG_WAIT_MS } from "../useOnboardingLayout";
import { makeProgress } from "./progress-fixture";

const BACKEND = "http://localhost:18006";
const USER_ID = "interview-user";
const SESSION_ID = "interview-session-1";
const INTRO_PATH_KEY = "autogpt:onboarding-intro-path";
const GREETING = "Hi, I'm Otto, an AI. What do you do?";
const KICKOFF = "The user has opened the onboarding interview. Start it now.";

// The stage is read for a moment after each turn; shortened here so a test
// waits on the read itself, not on the clock.
vi.mock("../steps/InterviewStep/helpers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../steps/InterviewStep/helpers")>()),
  STAGE_READ_INTERVAL_MS: 50,
  STAGE_READ_WINDOW_MS: 400,
}));

vi.mock("@/services/environment", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      isLocal: () => false,
      getAGPTServerBaseUrl: () => BACKEND,
    },
  };
});

vi.mock("@/app/(platform)/copilot/helpers", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/app/(platform)/copilot/helpers")
  >()),
  getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
}));

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));

vi.mock("posthog-js", () => ({
  default: { capture: vi.fn() },
}));

vi.mock("@/components/atoms/SwapFade/SwapFade", () => ({
  SwapFade: function SwapFade({
    children,
    className,
  }: {
    children: ReactNode;
    className?: string;
  }) {
    return <div className={className}>{children}</div>;
  },
}));

vi.mock("../steps/BrainDumpStep/recordingStore", () => ({
  isIndexedDBAvailable: () => false,
  partId: (recordingId: string, partIndex: number) =>
    `${recordingId}:${partIndex}`,
  savePart: async () => undefined,
  markPartUploaded: async () => undefined,
  getParts: async () => [],
  saveMeta: async () => undefined,
  getMeta: async () => null,
  getMetaById: async () => null,
  clearRecording: async () => undefined,
  clearSkippedRecordings: async () => undefined,
}));

vi.mock("../steps/RoleStep", () => ({
  RoleStep: () => <div data-testid="step-role" />,
}));
vi.mock("../steps/SubscriptionStep/PlanTrialStep", () => ({
  PlanTrialStep: () => <div data-testid="step-plan" />,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams("step=interview"),
  usePathname: () => "/onboarding",
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    isLoggedIn: true,
    isUserLoading: false,
    user: { id: USER_ID, email: "interview@example.com", user_metadata: {} },
    refreshSession: vi.fn(() => Promise.resolve({ user: null })),
  }),
}));

// The step each save of the wizard's progress put the person on.
const savedSteps = vi.hoisted(() => [] as string[]);

vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1OnboardingState: () =>
    Promise.resolve({
      status: 200,
      data: {
        userId: USER_ID,
        completedSteps: [],
        wizardRevision: 0,
        wizardProgress: makeProgress({
          currentStep: "interview",
          completedSteps: ["team", "autopilot", "role"],
        }),
      },
    }),
  getV1CheckIfOnboardingIsCompleted: () =>
    Promise.resolve({ status: 200, data: false }),
  patchV1UpdateOnboardingState: (body: {
    wizardProgress: { currentStep: string };
  }) => {
    savedSteps.push(body.wizardProgress.currentStep);
    return Promise.resolve({
      status: 200,
      data: { userId: USER_ID, wizardRevision: 1 },
    });
  },
  postV1CompleteOnboardingStep: () => Promise.resolve({ status: 200 }),
  postV1SubmitOnboardingProfile: () => Promise.resolve({ status: 200 }),
}));

vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  getGetSubscriptionStatusQueryKey: () => ["/api/credits/subscription"],
  useGetSubscriptionStatus: (opts: {
    query: { select: (res: { status: number; data: unknown }) => unknown };
  }) => ({
    data: opts.query.select({ status: 200, data: { tier: "NO_TIER" } }),
    isLoading: false,
  }),
}));

vi.mock("@/app/api/helpers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/app/api/helpers")>()),
  resolveResponse: (p: Promise<{ data: unknown }>) => p.then((r) => r.data),
}));

let flags: Record<string, boolean> = {};
// While set, the page reads the interview flag's status from the flag service
// as it does in the app, so a test can hold the service's answer.
let readsFlagService = false;
vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  function servedFlagStatus(flag: string) {
    return { enabled: flags[flag] ?? false, ready: true, answered: true };
  }
  return {
    ...actual,
    useGetFlag: (flag: string) => flags[flag] ?? false,
    get useFlagStatus() {
      return readsFlagService ? actual.useFlagStatus : servedFlagStatus;
    },
  };
});

type SessionMessages = NonNullable<SessionDetailResponse["messages"]>;

function row(
  sequence: number,
  role: "user" | "assistant",
  content: string,
  metadata: Record<string, unknown> | null = null,
): SessionMessages[number] {
  return {
    id: `db-${sequence}`,
    role,
    content,
    tool_call_id: null,
    tool_calls: null,
    sequence,
    duration_ms: null,
    created_at: `2026-11-09T00:${String(sequence).padStart(2, "0")}:00Z`,
    metadata,
  };
}

const KICKOFF_ROW = row(1, "user", KICKOFF, {
  hidden: true,
  kind: "onboarding_interview_kickoff",
});
const GREETING_ROW = row(2, "assistant", GREETING);
const ANSWER_ROW = row(3, "user", "I run a bakery with five staff.");
const SUMMARY =
  "So your team starts on the order emails in Gmail. Is that right?";
const SUMMARY_ROW = row(4, "assistant", SUMMARY);

/** The interview as the server holds it; tests move it along. */
interface ServerInterview {
  messages: SessionMessages;
  stage: string | null;
  activeTurn: string | null;
}

/** What one call to the finish route answers. */
interface FinishAnswer {
  status: number;
  body?: JsonBodyType;
  networkError?: boolean;
}

const HANDED_OFF: FinishAnswer = {
  status: 200,
  body: { status: "transcribed", input_mode: "typed" },
};

let serverInterview: ServerInterview;
let streamBodies: string[];
let pendingBodies: string[];
let finishCalls: number;
let finishAnswers: FinishAnswer[];
let skipBodies: unknown[];
let stageReads: number;
let replyHold: ReturnType<typeof createStreamHold> | null;

function session(): SessionDetailResponse {
  return {
    id: SESSION_ID,
    created_at: "2026-11-09T00:00:00Z",
    updated_at: "2026-11-09T00:00:00Z",
    user_id: USER_ID,
    chat_status: serverInterview.activeTurn ? "running" : "idle",
    messages: serverInterview.messages,
    has_more_messages: false,
    oldest_sequence: null,
    active_stream: serverInterview.activeTurn
      ? {
          turn_id: serverInterview.activeTurn,
          last_message_id: "0-0",
          started_at: "2026-11-09T00:01:00Z",
        }
      : null,
    metadata: {
      dry_run: false,
      builder_graph_id: null,
      kind: "onboarding_interview",
      onboarding_interview_stage: serverInterview.stage,
    },
    expert_id: null,
    background_pending: false,
  };
}

/** The chat's server side: each turn is written to the interview's rows as
 *  the backend writes it, and Otto's reply may move the stage. */
function interviewHandlers({
  replyText = "Nice. Where does your week go?",
  stageAfterReply,
}: {
  replyText?: string;
  stageAfterReply?: string;
} = {}) {
  return [
    getGetV2GetSessionMockHandler200((info) => {
      // The step reads the stage alone, one row, after each turn.
      if (new URL(info.request.url).searchParams.get("limit") === "1") {
        stageReads += 1;
      }
      return session();
    }),
    getGetV2GetPendingMessagesMockHandler200({ messages: [], count: 0 }),
    http.get(
      `${BACKEND}/api/chat/sessions/${SESSION_ID}/stream`,
      ({ request }) => {
        serverInterview = {
          ...serverInterview,
          messages: [KICKOFF_ROW, GREETING_ROW],
          activeTurn: null,
        };
        return streamSseResponse(
          assistantTextChunks(GREETING, { messageId: "greeting" }),
          { abortSignal: request.signal },
        );
      },
    ),
    http.post(
      `${BACKEND}/api/chat/sessions/${SESSION_ID}/stream`,
      async ({ request }) => {
        const body = await request.clone().text();
        streamBodies.push(body);
        const said = (JSON.parse(body) as { message?: string }).message ?? "";
        const next = serverInterview.messages.length + 1;
        serverInterview = {
          ...serverInterview,
          messages: [
            ...serverInterview.messages,
            row(next, "user", said),
            row(next + 1, "assistant", replyText),
          ],
          stage: stageAfterReply ?? serverInterview.stage,
        };
        return streamSseResponse(assistantTextChunks(replyText), {
          abortSignal: request.signal,
          hold: replyHold?.hold,
        });
      },
    ),
    http.post(
      `${BACKEND}/api/chat/sessions/${SESSION_ID}/messages/pending`,
      async ({ request }) => {
        pendingBodies.push(await request.clone().text());
        return HttpResponse.json({
          buffer_length: pendingBodies.length,
          max_buffer_length: 10,
          turn_in_flight: true,
        });
      },
    ),
    http.post("/api/proxy/api/onboarding/brain-dump/interview/finish", () => {
      finishCalls += 1;
      const answer = finishAnswers.shift() ?? HANDED_OFF;
      if (answer.networkError) return HttpResponse.error();
      if (answer.status === 200) serverInterview.stage = "handed_off";
      return HttpResponse.json(answer.body ?? {}, { status: answer.status });
    }),
    getFinalizeBrainDumpMockHandler200(async (info) => {
      skipBodies.push(await info.request.json());
      return { status: "completed", input_mode: "skipped" };
    }),
  ];
}

function startsWith(response: {
  session_id: string;
  turn_id: string | null;
  stage: string | null;
}) {
  server.use(getStartOnboardingInterviewMockHandler200(response));
}

function startFailsWith(
  status: number,
  body: { detail: string | Record<string, unknown> },
) {
  server.use(
    http.post("/api/proxy/api/onboarding/brain-dump/interview", () =>
      HttpResponse.json(body, { status }),
    ),
  );
}

async function sendAnswer(text: string) {
  const user = userEvent.setup();
  const input = await screen.findByPlaceholderText("Type your answer…");
  await waitFor(() => expect(input).toHaveProperty("disabled", false));
  await user.type(input, text);
  await user.keyboard("{Enter}");
}

function expectPlanStep() {
  return waitFor(() => expect(screen.getByTestId("step-plan")).toBeDefined());
}

/** An interview Otto has wrapped up and closed, as start answers it. */
function returnsClosed(stage: "closed" | "handed_off" = "closed") {
  serverInterview = {
    messages: [KICKOFF_ROW, GREETING_ROW, ANSWER_ROW, SUMMARY_ROW],
    stage,
    activeTurn: null,
  };
  startsWith({ session_id: SESSION_ID, turn_id: null, stage });
  server.use(...interviewHandlers());
}

async function clickMeetYourTeam() {
  await userEvent
    .setup()
    .click(await screen.findByRole("button", { name: "Meet your team" }));
}

const backendSocket = ws.link("ws://localhost:8001/ws");

beforeEach(() => {
  flags = { [FrontendFlag.ONBOARDING_INTERVIEW]: true };
  readsFlagService = false;
  savedSteps.length = 0;
  serverInterview = {
    messages: [KICKOFF_ROW, GREETING_ROW],
    stage: null,
    activeTurn: null,
  };
  streamBodies = [];
  pendingBodies = [];
  finishCalls = 0;
  finishAnswers = [];
  skipBodies = [];
  stageReads = 0;
  replyHold = null;
  server.use(backendSocket.addEventListener("connection", () => {}));
  useAuthStore.setState({
    user: { id: USER_ID, email: "interview@example.com", user_metadata: {} },
    isUserLoading: false,
    hasLoadedUser: true,
  });
  useOnboardingWizardStore.getState().reset();
  // Every test talks to the same interview session; its chat must not carry
  // over from the test before.
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
  localStorage.clear();
  sessionStorage.clear();
});

afterEach(() => {
  cleanup();
  useAuthStore.setState({ user: null, hasLoadedUser: false });
});

const GOODBYE = "Thanks! Your team starts on the order emails first.";
const NEXT_QUESTION = "Nice. Where does your week go?";

describe("onboarding interview", () => {
  it("streams Otto's greeting first and takes the person's answer", async () => {
    serverInterview = {
      messages: [KICKOFF_ROW],
      stage: null,
      activeTurn: "turn-greeting",
    };
    startsWith({
      session_id: SESSION_ID,
      turn_id: "turn-greeting",
      stage: null,
    });
    server.use(...interviewHandlers());

    render(<OnboardingPage />);

    expect(await screen.findByText(GREETING)).toBeDefined();
    // The kickoff the server wrote is not something the person said.
    expect(screen.queryByText(KICKOFF)).toBeNull();

    await sendAnswer("I run a bakery with five staff.");

    expect(await screen.findByText(NEXT_QUESTION)).toBeDefined();
    expect(streamBodies.join()).toContain("I run a bakery with five staff.");
  });

  it("gives the composer's place to Meet your team once the interview closes, keeping the goodbye", async () => {
    serverInterview.messages = [
      KICKOFF_ROW,
      GREETING_ROW,
      ANSWER_ROW,
      SUMMARY_ROW,
    ];
    startsWith({ session_id: SESSION_ID, turn_id: null, stage: null });
    server.use(
      ...interviewHandlers({
        replyText: GOODBYE,
        // Otto's end signal closes the interview inside the goodbye turn.
        stageAfterReply: "closed",
      }),
    );
    // The goodbye is written, and the turn has not ended yet.
    replyHold = createStreamHold(5);

    render(<OnboardingPage />);
    await sendAnswer("Yes, that's right.");

    expect(await screen.findByText(GOODBYE)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Meet your team" })).toBeNull();

    replyHold.release();
    const meet = await screen.findByRole("button", { name: "Meet your team" });

    expect(screen.getByText(GOODBYE)).toBeDefined();
    expect(screen.queryByPlaceholderText("Type your answer…")).toBeNull();
    // Skip is for an open interview.
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();

    await userEvent.setup().click(meet);

    await expectPlanStep();
    expect(finishCalls).toBe(1);
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
  });

  it("shows Meet your team straight away for an interview already handed off", async () => {
    returnsClosed("handed_off");

    render(<OnboardingPage />);

    expect(
      await screen.findByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(screen.queryByPlaceholderText("Type your answer…")).toBeNull();
    expect(await screen.findByText(SUMMARY)).toBeDefined();
  });

  it("hides I'm done while Otto replies and offers it again once the turn ends with no close", async () => {
    serverInterview.messages = [KICKOFF_ROW, GREETING_ROW, ANSWER_ROW];
    startsWith({ session_id: SESSION_ID, turn_id: null, stage: null });
    server.use(...interviewHandlers());
    replyHold = createStreamHold(5);

    render(<OnboardingPage />);
    expect(
      await screen.findByRole("button", { name: "I'm done" }),
    ).toBeDefined();

    await sendAnswer("Mostly answering order emails.");
    expect(await screen.findByText(NEXT_QUESTION)).toBeDefined();
    expect(screen.queryByRole("button", { name: "I'm done" })).toBeNull();
    const readsBeforeTheTurnEnded = stageReads;

    replyHold.release();

    // Offered as soon as the turn is over, with no wait on the stage reads.
    expect(
      await screen.findByRole("button", { name: "I'm done" }),
    ).toBeDefined();
    // The stage is still read after the turn, and finds no end.
    await waitFor(() =>
      expect(stageReads).toBeGreaterThan(readsBeforeTheTurnEnded),
    );
    expect(screen.getByPlaceholderText("Type your answer…")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Meet your team" })).toBeNull();
  });

  it("queues a message sent while Otto is still replying", async () => {
    serverInterview.messages = [KICKOFF_ROW, GREETING_ROW];
    startsWith({ session_id: SESSION_ID, turn_id: null, stage: null });
    server.use(...interviewHandlers());
    replyHold = createStreamHold(5);

    render(<OnboardingPage />);
    await sendAnswer("I run a bakery.");
    expect(await screen.findByText(NEXT_QUESTION)).toBeDefined();

    await sendAnswer("We have five staff.");

    await waitFor(() => expect(pendingBodies).toHaveLength(1));
    expect(pendingBodies[0]).toContain("We have five staff.");
    expect(streamBodies).toHaveLength(1);
    replyHold.release();
  });

  it("finishes with I'm done and moves on as path A", async () => {
    serverInterview.messages = [KICKOFF_ROW, GREETING_ROW, ANSWER_ROW];
    startsWith({ session_id: SESSION_ID, turn_id: null, stage: null });
    server.use(...interviewHandlers());

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "I'm done" }));

    await expectPlanStep();
    expect(finishCalls).toBe(1);
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
  });

  it("does not offer I'm done before the person has answered", async () => {
    startsWith({ session_id: SESSION_ID, turn_id: null, stage: null });
    server.use(...interviewHandlers());

    render(<OnboardingPage />);

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(screen.queryByRole("button", { name: "I'm done" })).toBeNull();
  });

  it("returns to a finished interview as its transcript and the finish button, with nothing finished until the click", async () => {
    returnsClosed();

    render(<OnboardingPage />);

    expect(
      await screen.findByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(await screen.findByText(SUMMARY)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
    expect(finishCalls).toBe(0);

    await clickMeetYourTeam();

    await expectPlanStep();
    expect(finishCalls).toBe(1);
  });

  it("goes Back from the plan step to the finished interview, and Back again to the role step", async () => {
    returnsClosed();
    const user = userEvent.setup();

    render(<OnboardingPage />);
    await clickMeetYourTeam();
    await expectPlanStep();

    startsWith({ session_id: SESSION_ID, turn_id: null, stage: "handed_off" });
    await user.click(screen.getByRole("button", { name: "Back" }));

    expect(
      await screen.findByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(await screen.findByText(SUMMARY)).toBeDefined();
    expect(finishCalls).toBe(1);

    await user.click(screen.getByRole("button", { name: "Back" }));

    expect(await screen.findByTestId("step-role")).toBeDefined();
  });

  it.each([["insufficient_content"], ["no_usable_speech"]])(
    "moves on as path B, with no skipped take, when the judge rules %s",
    async (code) => {
      returnsClosed();
      finishAnswers = [
        {
          status: 200,
          body: { status: "failed", input_mode: "typed", error_code: code },
        },
      ];

      render(<OnboardingPage />);
      await clickMeetYourTeam();

      await expectPlanStep();
      expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("B");
      expect(skipBodies).toEqual([]);
    },
  );

  it("moves on as path A when a newer take owns the row", async () => {
    returnsClosed();
    finishAnswers = [
      {
        status: 200,
        body: {
          status: "failed",
          input_mode: "typed",
          error_code: "superseded",
        },
      },
    ];

    render(<OnboardingPage />);
    await clickMeetYourTeam();

    await expectPlanStep();
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
  });

  it.each([["quality_check_unavailable"], ["something_else"], [null]])(
    "stays with Try again when the pipeline never ruled (%s), and the next click finishes again",
    async (code) => {
      returnsClosed();
      finishAnswers = [
        {
          status: 200,
          body: { status: "failed", input_mode: "typed", error_code: code },
        },
        HANDED_OFF,
      ];

      render(<OnboardingPage />);
      await clickMeetYourTeam();

      expect(
        await screen.findByText("That didn't go through. Try again."),
      ).toBeDefined();
      expect(screen.queryByTestId("step-plan")).toBeNull();
      expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBeNull();

      await clickMeetYourTeam();

      await expectPlanStep();
      expect(finishCalls).toBe(2);
      expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
    },
  );

  it("says how long to wait when the finish is turned away by the budget", async () => {
    returnsClosed();
    finishAnswers = [
      {
        status: 429,
        body: {
          detail: {
            message: "You have made several personalization attempts.",
            retry_after_seconds: 1200,
          },
        },
      },
    ];

    render(<OnboardingPage />);
    await clickMeetYourTeam();

    expect(
      await screen.findByText(
        "You've sent a lot of messages in a short time. Your progress is saved; try again in 20 minutes.",
      ),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(screen.queryByTestId("step-plan")).toBeNull();
  });

  it.each([
    ["the service is down", { status: 503, body: { detail: "Unavailable" } }],
    ["the request never gets an answer", { status: 0, networkError: true }],
  ])("stays with Try again when %s", async (_name, answer: FinishAnswer) => {
    returnsClosed();
    finishAnswers = [answer];

    render(<OnboardingPage />);
    await clickMeetYourTeam();

    expect(
      await screen.findByText("That didn't go through. Try again."),
    ).toBeDefined();
    expect(screen.queryByTestId("step-plan")).toBeNull();
  });

  it("shows Try again when the start is refused with a 409", async () => {
    server.use(...interviewHandlers());
    startFailsWith(409, { detail: "turn_in_progress" });

    render(<OnboardingPage />);

    expect(
      await screen.findByText("We couldn't start your interview. Try again."),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Try again" })).toBeDefined();
    expect(finishCalls).toBe(0);
  });

  it("says how long to wait when the interview's turn budget is spent", async () => {
    server.use(...interviewHandlers());
    startFailsWith(429, {
      detail: {
        message: "You have sent a lot of messages in a short time.",
        retry_after_seconds: 1200,
      },
    });

    render(<OnboardingPage />);

    expect(
      await screen.findByText(
        "You've sent a lot of messages in a short time. Your progress is saved; try again in 20 minutes.",
      ),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Try again" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Skip for now" })).toBeDefined();
  });

  it("starts again from Try again once the interview is back", async () => {
    server.use(...interviewHandlers());
    startFailsWith(503, {
      detail: "The interview is temporarily unavailable.",
    });

    render(<OnboardingPage />);
    const retry = await screen.findByRole("button", { name: "Try again" });
    expect(screen.getByText(/Otto can't chat right now/)).toBeDefined();
    startsWith({ session_id: SESSION_ID, turn_id: null, stage: null });

    await userEvent.setup().click(retry);

    expect(await screen.findByText(GREETING)).toBeDefined();
  });

  it("keeps Skip as today's skip: path B, with nothing handed off", async () => {
    startsWith({ session_id: SESSION_ID, turn_id: null, stage: null });
    server.use(...interviewHandlers());

    render(<OnboardingPage />);
    await screen.findByText(GREETING);
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Skip for now" }));

    await expectPlanStep();
    await waitFor(() =>
      expect(skipBodies).toEqual([
        expect.objectContaining({ input_mode: "skipped" }),
      ]),
    );
    expect(finishCalls).toBe(0);
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("B");
  });

  it("does nothing on skip once the interview is over", async () => {
    returnsClosed();
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const { result } = renderHook(() => useInterviewStep(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    });
    await waitFor(() => expect(result.current.isOver).toBe(true));
    const stepBefore = useOnboardingWizardStore.getState().currentStep;

    await act(async () => {
      await result.current.skip();
    });

    expect(useOnboardingWizardStore.getState().currentStep).toBe(stepBefore);
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBeNull();
    expect(skipBodies).toEqual([]);
    expect(finishCalls).toBe(0);
  });

  it("shows Meet your team when a message is refused because the interview is over", async () => {
    serverInterview.messages = [KICKOFF_ROW, GREETING_ROW, ANSWER_ROW];
    startsWith({ session_id: SESSION_ID, turn_id: null, stage: null });
    // The first handler that matches answers, so the refusal goes first.
    server.use(
      http.post(`${BACKEND}/api/chat/sessions/${SESSION_ID}/stream`, () =>
        HttpResponse.json(
          { detail: "onboarding_interview_closed" },
          { status: 409 },
        ),
      ),
      ...interviewHandlers(),
    );

    render(<OnboardingPage />);
    await sendAnswer("One more thing.");

    expect(
      await screen.findByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
  });

  it("keeps the brain dump while the interview flag is off", async () => {
    flags = {};
    let starts = 0;
    server.use(
      http.post("/api/proxy/api/onboarding/brain-dump/interview", () => {
        starts += 1;
        return HttpResponse.json({});
      }),
    );

    render(<OnboardingPage />);

    expect(await screen.findByText("Talk to me about your work")).toBeDefined();
    expect(screen.queryByText("Tell Otto about your work")).toBeNull();
    expect(starts).toBe(0);
  });
});

describe("onboarding interview, with a slow flag service", () => {
  // The flag service answers the interview flag on, once the test lets it.
  function holdFlagService() {
    let answer = () => undefined as void;
    const answered = new Promise<void>((resolve) => {
      answer = resolve;
    });
    server.use(
      http.get("*/api/feature-flags", async () => {
        await answered;
        return HttpResponse.json({
          configured: true,
          flags: { [FrontendFlag.ONBOARDING_INTERVIEW]: true },
        });
      }),
    );
    return answer;
  }

  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    readsFlagService = true;
    startsWith({ session_id: SESSION_ID, turn_id: null, stage: null });
    server.use(...interviewHandlers());
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it.each([
    // Past the flag's own 5 s timeout.
    [5_500],
    // Just under onboarding's own wait.
    [9_500],
  ])(
    "waits for the flag's answer at %d ms, and opens the interview the person saved",
    async (answeredAtMs) => {
      expect(answeredAtMs).toBeLessThan(INTERVIEW_FLAG_WAIT_MS);
      const answer = holdFlagService();
      render(<OnboardingPage />);

      await act(() => vi.advanceTimersByTimeAsync(answeredAtMs));
      expect(screen.queryByText("Talk to me about your work")).toBeNull();
      expect(screen.queryByText("Tell Otto about your work")).toBeNull();

      act(() => answer());

      expect(
        await screen.findByText("Tell Otto about your work"),
      ).toBeDefined();
      await act(() => vi.advanceTimersByTimeAsync(1_000));
      expect(savedSteps).not.toContain("painPoints");
    },
  );

  it("goes on without the interview for this visit once the flag has not answered in time, saving none of the brain dump's place", async () => {
    const answer = holdFlagService();
    render(<OnboardingPage />);

    await act(() => vi.advanceTimersByTimeAsync(INTERVIEW_FLAG_WAIT_MS + 500));

    expect(await screen.findByText("Talk to me about your work")).toBeDefined();
    await act(() => vi.advanceTimersByTimeAsync(1_000));
    expect(savedSteps).not.toContain("painPoints");
    expect(readLocalProgress(USER_ID)?.progress.currentStep).not.toBe(
      "painPoints",
    );
    act(() => answer());
  });
});
