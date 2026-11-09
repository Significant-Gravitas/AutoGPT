import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import { useCopilotStreamStore } from "@/app/(platform)/copilot/copilotStreamStore";
import {
  getEndOnboardingVoiceCallMockHandler202,
  getFinalizeBrainDumpMockHandler200,
  getFinishOnboardingVoiceCallMockHandler200,
  getStartOnboardingInterviewMockHandler200,
  getStartOnboardingVoiceCallMockHandler200,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import {
  getGetV2GetPendingMessagesMockHandler200,
  getGetV2GetSessionMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { FinalizeResponse } from "@/app/api/__generated__/models/finalizeResponse";
import { FrontendFlag } from "@/app/api/__generated__/models/frontendFlag";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { VoiceCallStartResponse } from "@/app/api/__generated__/models/voiceCallStartResponse";
import type { VoiceCallStatusResponse } from "@/app/api/__generated__/models/voiceCallStatusResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import { capturePostHogEvent } from "@/services/analytics/posthog-capture";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import {
  act,
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  type FakeWebSocket,
  installVoiceFakes,
  RELAY_URL,
  type VoiceFakes,
} from "@/tests/integrations/voice-fakes";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse, type JsonBodyType, ws } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import OnboardingPage from "../page";
import { useOnboardingWizardStore } from "../store";
import { makeProgress } from "./progress-fixture";

const BACKEND = "http://localhost:18006";
const USER_ID = "voice-user";
const SESSION_ID = "voice-session-1";
const CALL_ID = "a".repeat(32);
const OTHER_CALL_ID = "b".repeat(32);
const TOKEN = "start-token";
const TICKET = "resume-ticket";
const INTRO_PATH_KEY = "autogpt:onboarding-intro-path";
const TICKET_KEY = `autogpt:onboarding-voice-call:${USER_ID}`;
const GREETING = "Hi, I'm Otto, an AI. What do you do?";
const VOICE = "/api/proxy/api/onboarding/brain-dump/interview/voice";
const CAPS = {
  max_call_s: 300,
  soft_wrap_s: 30,
  max_agent_turns: 16,
  soft_wrap_turns: 2,
  idle_s: 20,
};

// The call's waits, shortened so a test waits on what the step does, not on
// the clock.
vi.mock(
  "../steps/InterviewStep/components/VoiceCall/helpers",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("../steps/InterviewStep/components/VoiceCall/helpers")
    >()),
    REATTACH_WINDOW_MS: 2_000,
    REATTACH_RETRY_MS: 20,
    ENDING_POLL_MS: 20,
    ENDING_WINDOW_MS: 300,
    ENDING_SLOW_POLL_MS: 40,
    END_POLL_MS: 20,
    END_WAIT_MS: 500,
    START_RETRY_MS: 20,
  }),
);

vi.mock("../steps/InterviewStep/helpers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../steps/InterviewStep/helpers")>()),
  STAGE_READ_INTERVAL_MS: 50,
  STAGE_READ_WINDOW_MS: 400,
}));

vi.mock("@/services/analytics/posthog-capture", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/analytics/posthog-capture")
  >()),
  capturePostHogEvent: vi.fn(),
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
    user: { id: USER_ID, email: "voice@example.com", user_metadata: {} },
    refreshSession: vi.fn(() => Promise.resolve({ user: null })),
  }),
}));

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
  patchV1UpdateOnboardingState: () =>
    Promise.resolve({
      status: 200,
      data: { userId: USER_ID, wizardRevision: 1 },
    }),
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
vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: (flag: string) => flags[flag] ?? false,
  useFlagStatus: (flag: string) => ({
    enabled: flags[flag] ?? false,
    ready: true,
    answered: true,
  }),
}));

// --- The server's side -------------------------------------------------------

// A call's record as the call-state routes answer it, with its word on who
// owns the interview.
type CallRecord = VoiceCallStatusResponse & {
  interview?: "taken" | "superseded" | null;
};

interface Answer {
  status: number;
  body?: JsonBodyType;
}

const ACCEPTED: FinalizeResponse = {
  status: "transcribed",
  input_mode: "typed",
};

function callRecord(overrides: Partial<CallRecord> = {}): CallRecord {
  return {
    call_id: CALL_ID,
    status: "handed_off",
    end_reason: "finished",
    leg: "websocket",
    hand_off: ACCEPTED,
    ...overrides,
  };
}

const STARTED: VoiceCallStartResponse = {
  call_id: CALL_ID,
  relay_url: RELAY_URL,
  token: TOKEN,
  expires_in: 60,
  leg: { kind: "websocket" },
  caps: CAPS,
};

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

const KICKOFF_ROW = row(
  1,
  "user",
  "The user has opened the onboarding interview.",
  { hidden: true, kind: "onboarding_interview_kickoff" },
);
const GREETING_ROW = row(2, "assistant", GREETING);
const TYPED_ANSWER_ROW = row(3, "user", "I run a bakery with five staff.");

let latestCall: CallRecord | null;
let callStates: Record<string, (CallRecord | "missing")[]>;
let typedMessages: SessionMessages;
let turnAnswer: Answer | null;
let mints: number;
let latestReads: number;
let statusReads: string[];
let finishes: string[];
let ends: string[];
let typedStartCount: number;
let skipBodies: unknown[];
let requests: string[];
let fakes: VoiceFakes;

function session(): SessionDetailResponse {
  return {
    id: SESSION_ID,
    created_at: "2026-11-09T00:00:00Z",
    updated_at: "2026-11-09T00:00:00Z",
    user_id: USER_ID,
    chat_status: "idle",
    messages: typedMessages,
    has_more_messages: false,
    oldest_sequence: null,
    active_stream: null,
    metadata: {
      dry_run: false,
      builder_graph_id: null,
      kind: "onboarding_interview",
      onboarding_interview_stage: null,
    },
    expert_id: null,
    background_pending: false,
  };
}

function handlers() {
  return [
    // The person's latest call, read when the step opens.
    http.get(VOICE, () => {
      latestReads += 1;
      if (latestCall) return HttpResponse.json(latestCall);
      return HttpResponse.json(
        { detail: "No such voice call" },
        { status: 404 },
      );
    }),
    getStartOnboardingVoiceCallMockHandler200(() => {
      mints += 1;
      return STARTED;
    }),
    http.get(`${VOICE}/:callId`, ({ params }) => {
      const callId = String(params.callId);
      statusReads.push(callId);
      const queue = callStates[callId] ?? [];
      const next = queue.length > 1 ? queue.shift() : queue[0];
      if (!next || next === "missing") {
        return HttpResponse.json(
          { detail: "No such voice call" },
          { status: 404 },
        );
      }
      return HttpResponse.json(next);
    }),
    getFinishOnboardingVoiceCallMockHandler200(({ params }) => {
      finishes.push(String(params.callId));
      return ACCEPTED;
    }),
    getEndOnboardingVoiceCallMockHandler202(
      ({ params }: { params: { callId?: unknown } }) => {
        const callId = String(params.callId);
        ends.push(callId);
        // The relay ends a call within a second of the request.
        callStates[callId] = [
          callRecord({ call_id: callId, end_reason: "superseded" }),
        ];
        return undefined;
      },
    ),
    getStartOnboardingInterviewMockHandler200(() => {
      typedStartCount += 1;
      return { session_id: SESSION_ID, turn_id: null, stage: null };
    }),
    getGetV2GetSessionMockHandler200(() => session()),
    getGetV2GetPendingMessagesMockHandler200({ messages: [], count: 0 }),
    http.post(
      `${BACKEND}/api/chat/sessions/${SESSION_ID}/stream`,
      async ({ request }) => {
        if (turnAnswer) {
          return HttpResponse.json(turnAnswer.body ?? {}, {
            status: turnAnswer.status,
          });
        }
        return streamSseResponse(assistantTextChunks("Tell me more."), {
          abortSignal: request.signal,
        });
      },
    ),
    getFinalizeBrainDumpMockHandler200(async (info) => {
      skipBodies.push(await info.request.json());
      return { status: "completed", input_mode: "skipped" };
    }),
  ];
}

// The call's start answers this once, ahead of the usual start.
function refuseCallStartOnce(answer: Answer) {
  server.use(
    http.post(
      VOICE,
      () => {
        mints += 1;
        return HttpResponse.json(answer.body ?? {}, { status: answer.status });
      },
      { once: true },
    ),
  );
}

function refuseTypedStartOnce(answer: Answer) {
  server.use(
    http.post(
      "/api/proxy/api/onboarding/brain-dump/interview",
      () => {
        typedStartCount += 1;
        return HttpResponse.json(answer.body ?? {}, { status: answer.status });
      },
      { once: true },
    ),
  );
}

function keepTicket(callId = CALL_ID) {
  sessionStorage.setItem(
    TICKET_KEY,
    JSON.stringify({
      callId,
      relayUrl: RELAY_URL,
      ticket: TICKET,
      caps: CAPS,
      startedAt: Date.now() - 30_000,
      turnsLeft: 12,
    }),
  );
}

// --- The person's side -------------------------------------------------------

async function relaySocketAfter(previous?: FakeWebSocket) {
  return waitFor(() => {
    const socket = fakes.socket();
    if (!socket || socket === previous) throw new Error("No new socket yet");
    return socket;
  });
}

// "Talk to Otto", and the relay's socket opening.
async function callOtto() {
  await userEvent
    .setup()
    .click(await screen.findByRole("button", { name: "Talk to Otto" }));
  const socket = await relaySocketAfter();
  act(() => socket.accept());
  return socket;
}

async function liveCall() {
  const socket = await callOtto();
  act(() =>
    socket.relay({
      type: "ready",
      call_id: CALL_ID,
      caps: CAPS,
      resume: TICKET,
      leg: "websocket",
    }),
  );
  await screen.findByText("You're on the call with Otto");
  return socket;
}

function endCall(socket: FakeWebSocket, reason: string) {
  act(() => {
    socket.relay({ type: "ended", reason, elapsed_s: 92.4 });
    socket.close(1000);
  });
}

function expectPlanStep() {
  return waitFor(() => expect(screen.getByTestId("step-plan")).toBeDefined());
}

function tracked(event: string) {
  return vi
    .mocked(capturePostHogEvent)
    .mock.calls.filter(([name]) => name === event)
    .map(([, properties]) => properties);
}

const backendSocket = ws.link("ws://localhost:8001/ws");

function onRequest({ request }: { request: Request }) {
  requests.push(new URL(request.url).pathname);
}

beforeEach(() => {
  flags = {
    [FrontendFlag.ONBOARDING_INTERVIEW]: true,
    [FrontendFlag.ONBOARDING_INTERVIEW_VOICE]: true,
  };
  latestCall = null;
  callStates = {};
  typedMessages = [KICKOFF_ROW, GREETING_ROW];
  turnAnswer = null;
  mints = 0;
  latestReads = 0;
  statusReads = [];
  finishes = [];
  ends = [];
  typedStartCount = 0;
  skipBodies = [];
  requests = [];
  vi.mocked(capturePostHogEvent).mockClear();
  fakes = installVoiceFakes();
  server.use(
    backendSocket.addEventListener("connection", () => {}),
    ...handlers(),
  );
  server.events.on("request:start", onRequest);
  useAuthStore.setState({
    user: { id: USER_ID, email: "voice@example.com", user_metadata: {} },
    isUserLoading: false,
    hasLoadedUser: true,
  });
  useOnboardingWizardStore.getState().reset();
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
  localStorage.clear();
  sessionStorage.clear();
});

afterEach(() => {
  cleanup();
  server.events.removeListener("request:start", onRequest);
  fakes.uninstall();
  useAuthStore.setState({ user: null, hasLoadedUser: false });
  // The plan step is a stand-in here: nothing reaches the subscription or
  // its checkout from the interview step.
  expect(
    requests.filter((path) => /subscription|checkout|credits/.test(path)),
  ).toEqual([]);
});

describe("onboarding interview by voice: opening", () => {
  it("keeps PR 1's typed step while the voice flag is off", async () => {
    flags = { [FrontendFlag.ONBOARDING_INTERVIEW]: true };

    render(<OnboardingPage />);

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Talk to Otto" })).toBeNull();
    expect(latestReads).toBe(0);
    expect(mints).toBe(0);
  });

  it("opens with Talk to Otto and Type instead, with nothing typed started", async () => {
    render(<OnboardingPage />);

    expect(
      await screen.findByRole("button", { name: "Talk to Otto" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Type instead" })).toBeDefined();
    expect(
      screen.getByText(
        "Private to you · Saved as Otto memory · Downloadable anytime",
      ),
    ).toBeDefined();
    expect(latestReads).toBe(1);
    expect(typedStartCount).toBe(0);
  });

  it("types instead from the choice", async () => {
    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Type instead" }));

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(mints).toBe(0);
  });

  it("opens the typed interview with the notice when the browser has denied the microphone", async () => {
    fakes.denyMicrophonePermission();

    render(<OnboardingPage />);

    expect(
      await screen.findByText(
        "Your browser blocked the microphone, so write here instead.",
      ),
    ).toBeDefined();
    expect(await screen.findByText(GREETING)).toBeDefined();
    // The latest call is still read first.
    expect(latestReads).toBe(1);
  });

  it("moves to the typed interview with the notice when the microphone is refused on the call", async () => {
    fakes.refuseMicrophone("NotAllowedError");

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Talk to Otto" }));

    expect(
      await screen.findByText(
        "Your browser blocked the microphone, so write here instead.",
      ),
    ).toBeDefined();
    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(mints).toBe(0);
    expect(tracked("voice_interview_mic_denied")).toEqual([{}]);
  });

  it("falls to the typed interview without a word when calls are off for the person", async () => {
    refuseCallStartOnce({
      status: 404,
      body: { detail: "Feature not available" },
    });

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Talk to Otto" }));

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(screen.queryByText(/couldn't start the call/)).toBeNull();
    expect(fakes.sockets).toEqual([]);
  });

  it("says when calls can be made again once the allowance is spent", async () => {
    refuseCallStartOnce({
      status: 429,
      body: {
        detail: {
          message: "You have made several calls.",
          retry_after_seconds: 1200,
        },
      },
    });

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Talk to Otto" }));

    expect(
      await screen.findByText(
        "You've made a few calls already. You can call again in 20 minutes, or type instead.",
      ),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Type instead" })).toBeDefined();
  });

  it("offers Rejoin and End that call when a call of the person's is live", async () => {
    keepTicket(OTHER_CALL_ID);
    refuseCallStartOnce({
      status: 409,
      body: { detail: { code: "call_in_progress", call_id: OTHER_CALL_ID } },
    });

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Talk to Otto" }));

    expect(await screen.findByRole("button", { name: "Rejoin" })).toBeDefined();
    expect(screen.getByRole("button", { name: "End that call" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
  });

  it("ends the live call and starts a new one from End that call", async () => {
    refuseCallStartOnce({
      status: 409,
      body: { detail: { code: "call_in_progress", call_id: OTHER_CALL_ID } },
    });
    callStates[OTHER_CALL_ID] = [
      callRecord({ call_id: OTHER_CALL_ID, status: "live", end_reason: null }),
    ];

    render(<OnboardingPage />);
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Talk to Otto" }),
    );
    const endThatCall = await screen.findByRole("button", {
      name: "End that call",
    });
    // No ticket in this tab: the call is on another device.
    expect(screen.queryByRole("button", { name: "Rejoin" })).toBeNull();
    expect(
      screen.getByText(
        "Your call with Otto is still going in another tab or on another device.",
      ),
    ).toBeDefined();
    await user.click(endThatCall);

    const socket = await relaySocketAfter();
    act(() => socket.accept());
    expect(ends).toEqual([OTHER_CALL_ID]);
    expect(mints).toBe(2);
    expect(socket.messages()[0]).toEqual({
      type: "start",
      token: TOKEN,
      leg: "websocket",
    });
  });
});

describe("onboarding interview by voice: the call", () => {
  it("starts the clock and the turns when the call goes live", async () => {
    render(<OnboardingPage />);
    const socket = await callOtto();
    expect(socket.url).toBe(RELAY_URL);
    expect(socket.messages()[0]).toEqual({
      type: "start",
      token: TOKEN,
      leg: "websocket",
    });

    act(() =>
      socket.relay({
        type: "ready",
        call_id: CALL_ID,
        caps: CAPS,
        resume: TICKET,
        leg: "websocket",
      }),
    );

    expect(await screen.findByText("5:00 left")).toBeDefined();
    expect(screen.getByText("16 turns left")).toBeDefined();
    expect(tracked("interview_started")).toEqual([
      { mode: "voice", leg: "websocket" },
    ]);
  });

  it("sends the microphone's audio only once the call is live", async () => {
    render(<OnboardingPage />);
    const socket = await callOtto();

    act(() => fakes.speak());
    expect(socket.frames()).toHaveLength(0);

    act(() =>
      socket.relay({
        type: "ready",
        call_id: CALL_ID,
        caps: CAPS,
        resume: TICKET,
        leg: "websocket",
      }),
    );
    act(() => fakes.speak());

    expect(socket.frames()).toHaveLength(1);
  });

  it("captions what Otto and the person say", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();

    act(() => {
      socket.relay({ type: "agent_text", text: "Hi, I'm Otto." });
      socket.relay({ type: "agent_text", text: " What do you do?" });
      socket.relay({ type: "turn_complete", turn: 0, turns_left: 15 });
      socket.relay({ type: "user_text", text: "I run a bakery." });
    });

    expect(
      await screen.findByText("Hi, I'm Otto. What do you do?"),
    ).toBeDefined();
    expect(screen.getByText("I run a bakery.")).toBeDefined();
  });

  it("counts down Otto's turns as they complete", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();

    act(() => socket.relay({ type: "turn_complete", turn: 0, turns_left: 15 }));

    expect(await screen.findByText("15 turns left")).toBeDefined();
  });

  it("plays Otto's audio and drops what is queued when the person talks over it", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();
    const player = fakes.playerNode()!;

    act(() => socket.relayAudio());
    expect(player.port.postMessage).toHaveBeenLastCalledWith(
      expect.any(ArrayBuffer),
      [expect.any(ArrayBuffer)],
    );

    act(() => socket.relay({ type: "interrupted" }));
    expect(player.port.postMessage).toHaveBeenLastCalledWith({
      type: "flush",
    });
  });

  it("re-attaches a dropped call with its ticket, showing Reconnecting then Reconnected", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();

    act(() => socket.drop(1006));
    expect(await screen.findByText("Reconnecting…")).toBeDefined();

    const again = await relaySocketAfter(socket);
    act(() => again.accept());
    expect(again.messages()[0]).toEqual({ type: "resume", ticket: TICKET });
    act(() => again.relay({ type: "resumed", gap_s: 1.2 }));

    expect(await screen.findByText("Reconnected")).toBeDefined();
    expect(tracked("voice_interview_reconnected")).toEqual([{ gap_s: 1.2 }]);
  });

  it("hangs up only from Hang up, and sends nothing on pagehide", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();

    act(() => {
      window.dispatchEvent(new Event("pagehide"));
    });
    expect(socket.messages().filter((m) => m.type === "hangup")).toEqual([]);

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Hang up" }));

    expect(socket.messages().filter((m) => m.type === "hangup")).toEqual([
      { type: "hangup" },
    ]);
  });

  it("rejoins a call still live after a reload, with the ticket kept in this tab", async () => {
    keepTicket();
    latestCall = callRecord({
      status: "live",
      end_reason: null,
      hand_off: null,
    });

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Rejoin" }));

    const socket = await relaySocketAfter();
    act(() => socket.accept());
    expect(socket.messages()[0]).toEqual({ type: "resume", ticket: TICKET });
    act(() => socket.relay({ type: "resumed", gap_s: 4.1 }));

    expect(await screen.findByText("Reconnected")).toBeDefined();
    expect(screen.getByText("12 turns left")).toBeDefined();
    expect(mints).toBe(0);
  });
});

describe("onboarding interview by voice: how a call ends", () => {
  it("shows Meet your team once a finished call is handed off, and moves on as path A", async () => {
    callStates[CALL_ID] = [
      callRecord({ status: "transcribing", hand_off: null }),
      callRecord({ status: "handed_off", interview: "taken" }),
    ];
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "finished");

    expect(await screen.findByText("Wrapping up…")).toBeDefined();
    const meet = await screen.findByRole("button", { name: "Meet your team" });
    expect(screen.getByText("That's everything Otto needed.")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
    expect(tracked("interview_ended")).toEqual([
      { mode: "voice", reason: "finished", elapsed_s: 92.4, turns: 0 },
    ]);

    await userEvent.setup().click(meet);

    await expectPlanStep();
    expect(finishes).toEqual([]);
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
    expect(tracked("team_screen_reached")).toEqual([
      { mode: "voice", path: "A" },
    ]);
  });

  it("offers Try again through the finish route when the pipeline never ruled on the words", async () => {
    callStates[CALL_ID] = [
      callRecord({
        status: "failed",
        hand_off: {
          status: "failed",
          input_mode: "typed",
          error_code: "quality_check_unavailable",
        },
      }),
    ];
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "finished");
    const tryAgain = await screen.findByRole("button", { name: "Try again" });
    expect(
      screen.getByText(
        "What you said is kept, but we couldn't finish setting up from it yet.",
      ),
    ).toBeDefined();
    // The call ran to its close: the one way on is to hand it off again.
    expect(screen.queryByRole("button", { name: "Call again" })).toBeNull();
    await userEvent.setup().click(tryAgain);

    await expectPlanStep();
    expect(finishes).toEqual([CALL_ID]);
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
  });

  it("shows the lost screen with its three ways on after a call that went quiet, and types from it", async () => {
    callStates[CALL_ID] = [callRecord({ end_reason: "idle" })];
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "idle");

    expect(
      await screen.findByText(
        "We didn't hear anything for a while, so the call ended.",
      ),
    ).toBeDefined();
    expect(screen.getByText("What you said so far is saved.")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Call again" })).toBeDefined();
    // The interview is still open behind a call that was cut off.
    expect(screen.getByRole("button", { name: "Skip for now" })).toBeDefined();

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Type instead" }));

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(typedStartCount).toBe(1);
  });

  it("finishes a call that was cut off before Meet your team moves on", async () => {
    callStates[CALL_ID] = [callRecord({ end_reason: "vendor_error" })];
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "vendor_error");
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Meet your team" }));

    await expectPlanStep();
    expect(finishes).toEqual([CALL_ID]);
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
  });

  it("reads a lost call's words before saying anything about typing, then shows the lost screen", async () => {
    const transcribing = callRecord({
      status: "transcribing",
      end_reason: "client_gone",
      hand_off: null,
    });
    callStates[CALL_ID] = [
      callRecord({ status: "live", end_reason: null, hand_off: null }),
      ...Array.from({ length: 6 }, () => transcribing),
      callRecord({ end_reason: "client_gone" }),
    ];
    render(<OnboardingPage />);
    const socket = await liveCall();

    act(() => socket.drop(1006));
    const again = await relaySocketAfter(socket);
    // The relay holds no call for the ticket any more.
    act(() => again.drop(4404));

    expect(
      await screen.findByText("We lost the call. Saving what you said…"),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Type instead" })).toBeNull();

    expect(await screen.findByText("The connection dropped.")).toBeDefined();
    expect(screen.getByText("What you said so far is saved.")).toBeDefined();
    expect(statusReads.length).toBeGreaterThanOrEqual(8);
    expect(screen.getByRole("button", { name: "Type instead" })).toBeDefined();
  });

  it("offers typing, another call or skipping when the call's words said too little", async () => {
    callStates[CALL_ID] = [
      callRecord({
        status: "failed",
        hand_off: {
          status: "failed",
          input_mode: "typed",
          error_code: "no_usable_speech",
        },
      }),
    ];
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "finished");

    expect(
      await screen.findByText(/We heard you, but there wasn't enough/),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Type instead" })).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Continue without it" }),
    ).toBeDefined();

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Call again" }));

    await relaySocketAfter(socket);
    expect(mints).toBe(2);
  });

  it.each([["transcribing"], ["handing_off"]])(
    "keeps Wrapping up beside Type instead past the wait while the call is %s, and stops reading the call once the person types",
    async (status) => {
      callStates[CALL_ID] = [
        callRecord({
          status: status as CallRecord["status"],
          hand_off: null,
        }),
      ];
      render(<OnboardingPage />);
      const socket = await liveCall();

      endCall(socket, "finished");
      const typeInstead = await screen.findByRole("button", {
        name: "Type instead",
      });
      expect(screen.getByText("Wrapping up…")).toBeDefined();

      await userEvent.setup().click(typeInstead);
      expect(await screen.findByText(GREETING)).toBeDefined();
      const readsWhenTyping = statusReads.length;
      await new Promise((resolve) => setTimeout(resolve, 200));

      expect(statusReads.length).toBe(readsWhenTyping);
    },
  );
});

describe("onboarding interview by voice: one interview, typed or spoken", () => {
  it("shows the ending of the call the typed start names as the interview's take", async () => {
    flags = { [FrontendFlag.ONBOARDING_INTERVIEW]: true };
    refuseTypedStartOnce({
      status: 409,
      body: {
        detail: { code: "onboarding_interview_closed", call_id: CALL_ID },
      },
    });
    callStates[CALL_ID] = [callRecord({ interview: "taken" })];

    render(<OnboardingPage />);
    const meet = await screen.findByRole("button", { name: "Meet your team" });
    expect(screen.getByText("That's everything Otto needed.")).toBeDefined();
    await userEvent.setup().click(meet);

    await expectPlanStep();
    expect(finishes).toEqual([]);
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
  });

  it("moves on as path A with no finish when the call the typed start names has expired", async () => {
    flags = { [FrontendFlag.ONBOARDING_INTERVIEW]: true };
    refuseTypedStartOnce({
      status: 409,
      body: {
        detail: { code: "onboarding_interview_closed", call_id: CALL_ID },
      },
    });
    callStates[CALL_ID] = ["missing"];

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Meet your team" }));

    await expectPlanStep();
    expect(finishes).toEqual([]);
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
  });

  it("ends the live call from End that call and opens the typed interview again", async () => {
    flags = { [FrontendFlag.ONBOARDING_INTERVIEW]: true };
    refuseTypedStartOnce({
      status: 409,
      body: { detail: { code: "call_in_progress", call_id: OTHER_CALL_ID } },
    });
    callStates[OTHER_CALL_ID] = [
      callRecord({ call_id: OTHER_CALL_ID, status: "live", end_reason: null }),
    ];

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "End that call" }));

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(ends).toEqual([OTHER_CALL_ID]);
    expect(typedStartCount).toBe(2);
  });

  it("shows the call's ending, not an error on the chat, when a typed turn is refused for a call that took the interview", async () => {
    turnAnswer = {
      status: 409,
      body: {
        detail: { code: "onboarding_interview_closed", call_id: CALL_ID },
      },
    };
    callStates[CALL_ID] = [callRecord({ interview: "taken" })];
    render(<OnboardingPage />);
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Type instead" }),
    );
    const input = await screen.findByPlaceholderText("Type your answer…");
    await waitFor(() => expect(input).toHaveProperty("disabled", false));

    await user.type(input, "One more thing.");
    await user.keyboard("{Enter}");

    expect(
      await screen.findByText("That's everything Otto needed."),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(screen.queryByPlaceholderText("Type your answer…")).toBeNull();
    expect(screen.queryByText(/onboarding_interview_closed/)).toBeNull();
  });

  it("opens the typed interview with no call screen when a later take superseded the latest call", async () => {
    latestCall = callRecord({ interview: "superseded" });

    render(<OnboardingPage />);

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Meet your team" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Talk to Otto" })).toBeNull();
    expect(statusReads).toEqual([]);
  });

  it("opens the call's ending when the latest call took the interview, even with typed turns in the session", async () => {
    typedMessages = [KICKOFF_ROW, GREETING_ROW, TYPED_ANSWER_ROW];
    latestCall = callRecord({ end_reason: "client_gone", interview: "taken" });

    render(<OnboardingPage />);

    expect(
      await screen.findByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Call again" })).toBeNull();
    expect(screen.queryByText(GREETING)).toBeNull();
    // The typed interview is never read first.
    expect(typedStartCount).toBe(0);
  });

  it("keeps Skip as path B, with nothing handed off, while the interview is open", async () => {
    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Skip for now" }));

    await expectPlanStep();
    await waitFor(() =>
      expect(skipBodies).toEqual([
        expect.objectContaining({ input_mode: "skipped" }),
      ]),
    );
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("B");
    expect(mints).toBe(0);
  });

  it("drops Skip once a call has taken the interview", async () => {
    latestCall = callRecord({ interview: "taken" });

    render(<OnboardingPage />);

    expect(
      await screen.findByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
  });
});
