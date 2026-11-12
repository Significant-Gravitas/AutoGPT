import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import { useCopilotStreamStore } from "@/app/(platform)/copilot/copilotStreamStore";
import {
  getEndOnboardingVoiceCallMockHandler202,
  getFinalizeBrainDumpMockHandler200,
  getFinishOnboardingVoiceCallMockHandler200,
  getGetOnboardingInterviewMockHandler200,
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
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
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
  DISCONNECT_REASON,
  type FakeRemoteTrack,
  FakeWebSocket,
  installVoiceFakes,
  RELAY_URL,
  type VoiceFakes,
} from "@/tests/integrations/voice-fakes";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse, type JsonBodyType, ws } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import OnboardingPage from "../page";
import { reportCallStart } from "../steps/InterviewStep/components/VoiceCall/callTimings";
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
// What a call's `interview_ended` says of Otto's speech when no stats of it
// were read.
const NOT_LISTENED = {
  concealed_share: null,
  silent_concealed_share: null,
  packets_lost: null,
  jitter_ms_max: null,
  rtt_ms_max: null,
};
// WebRTC's stats for Otto's speech in a call's room: 5% of his speech made
// up for packets that never came and 25% of his silence filled, 18 ms of
// jitter, and a 142 ms round trip on the candidate pair in use.
const OTTO_STATS = [
  {
    id: "IT01A1",
    type: "inbound-rtp",
    kind: "audio",
    concealedSamples: 28_800,
    silentConcealedSamples: 24_000,
    totalSamplesReceived: 96_000,
    packetsLost: 12,
    jitter: 0.018,
  },
  { id: "T01", type: "transport", selectedCandidatePairId: "CP01" },
  { id: "CP01", type: "candidate-pair", currentRoundTripTime: 0.142 },
  { id: "CP02", type: "candidate-pair", currentRoundTripTime: 0.9 },
];
const OTTO_HEARD = {
  concealed_share: 0.05,
  silent_concealed_share: 0.25,
  packets_lost: 12,
  jitter_ms_max: 18,
  rtt_ms_max: 142,
};
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
    HANGUP_FALLBACK_MS: 150,
    OPENING_READ_TIMEOUT_MS: 400,
    ROOM_JOIN_TIMEOUT_MS: 150,
    ROOM_PLAYOUT_MS: 300,
    OTTO_STATS_POLL_MS: 30,
  }),
);

vi.mock(
  "livekit-client",
  async () => (await import("@/tests/integrations/voice-fakes")).fakeLiveKit,
);

// The room's SDK fails to download while set.
let sdkFails = false;
vi.mock(
  "../steps/InterviewStep/components/VoiceCall/callRoom",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("../steps/InterviewStep/components/VoiceCall/callRoom")
      >();
    return {
      ...actual,
      loadLiveKit: () =>
        sdkFails
          ? Promise.reject(new Error("The chunk could not load"))
          : actual.loadLiveKit(),
    };
  },
);

vi.mock(
  "../steps/InterviewStep/components/VoiceCall/callTimings",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("../steps/InterviewStep/components/VoiceCall/callTimings")
    >()),
    reportCallStart: vi.fn(),
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

// A call's record as the call-state routes answer it.
type CallRecord = VoiceCallStatusResponse;

interface Answer {
  status: number;
  body?: JsonBodyType;
}

const ACCEPTED: FinalizeResponse = {
  status: "transcribed",
  input_mode: "typed",
};

// The finish's answer for a call whose take another take beat.
const SUPERSEDED_TAKE: FinalizeResponse = {
  status: "failed",
  input_mode: "typed",
  error_code: "superseded",
};

const DIDNT_HEAR_YOU =
  "We didn't hear you on that call. You can call again or type instead.";
const SAVE_FAILED =
  "Something went wrong saving that call. You can call again or type instead.";

function callRecord(overrides: Partial<CallRecord> = {}): CallRecord {
  return {
    call_id: CALL_ID,
    status: "handed_off",
    end_reason: "finished",
    hand_off: ACCEPTED,
    ...overrides,
  };
}

// The call screen's line on a call where Otto's voice reaches the microphone.
const HEARS_HIMSELF =
  "Otto can hear himself through your speakers, so he'll finish speaking before he listens. Headphones fix that.";

const ROOM_URL = "wss://agpt-test.livekit.cloud";
const ROOM_TOKEN = "room-token";
// A start: the relay's socket and the token for it, and the call's room and
// the browser's token for it.
const STARTED: VoiceCallStartResponse = {
  call_id: CALL_ID,
  relay_url: RELAY_URL,
  token: TOKEN,
  expires_in: 60,
  room: { url: ROOM_URL, token: ROOM_TOKEN },
  caps: CAPS,
};

// A live call as its reads answer: the relay's socket and the LiveKit
// project to rejoin it on.
function liveRoomRecord(overrides: Partial<CallRecord> = {}): CallRecord {
  return callRecord({
    status: "live",
    end_reason: null,
    hand_off: null,
    relay_url: RELAY_URL,
    livekit_url: ROOM_URL,
    ...overrides,
  });
}

// The person's typed interview as the step reads it: not opened yet, with a
// call on offer.
const NO_INTERVIEW = {
  session_id: null,
  stage: null,
  is_over: false,
  voice_available: true,
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
const TYPED_GOODBYE = "Thanks! Your team starts on the order emails first.";
const TYPED_GOODBYE_ROW = row(4, "assistant", TYPED_GOODBYE);

let latestCall: CallRecord | null;
// What the call's start answers.
let mintAnswer: VoiceCallStartResponse;
let callStates: Record<string, (CallRecord | "missing")[]>;
let typedMessages: SessionMessages;
let turnAnswer: Answer | null;
let mints: number;
// What each start asked for.
let mintBodies: unknown[];
let latestReads: number;
let statusReads: string[];
let finishes: string[];
let ends: string[];
let typedStartCount: number;
let typedReads: number;
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
    getStartOnboardingVoiceCallMockHandler200(async ({ request }) => {
      mints += 1;
      mintBodies.push(await request.text());
      return mintAnswer;
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
    http.get("/api/proxy/api/onboarding/brain-dump/interview", () => {
      typedReads += 1;
      return HttpResponse.json(NO_INTERVIEW);
    }),
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

function refuseTypedFinish(answer: Answer) {
  server.use(
    http.post("/api/proxy/api/onboarding/brain-dump/interview/finish", () =>
      HttpResponse.json(answer.body ?? {}, { status: answer.status }),
    ),
  );
}

// A ticket this tab kept, with this browser's token for the call's room, and
// anything else *extra* puts beside it.
function keepTicket(
  callId = CALL_ID,
  roomToken: string | null = ROOM_TOKEN,
  extra: Record<string, unknown> = {},
) {
  sessionStorage.setItem(
    TICKET_KEY,
    JSON.stringify({
      callId,
      ticket: TICKET,
      caps: CAPS,
      startedAt: Date.now() - 30_000,
      turnsLeft: 12,
      roomToken,
      ...extra,
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

async function talkToOtto() {
  await userEvent
    .setup()
    .click(await screen.findByRole("button", { name: "Talk to Otto" }));
}

// "Talk to Otto": the browser joins the call's room, and the relay's socket
// opens.
async function callOtto() {
  await talkToOtto();
  const socket = await relaySocketAfter();
  act(() => socket.accept());
  return socket;
}

function ready(socket: FakeWebSocket) {
  act(() =>
    socket.relay({
      type: "ready",
      call_id: CALL_ID,
      caps: CAPS,
      resume: TICKET,
    }),
  );
}

// A call gone live, in its room.
async function liveRoomCall() {
  const socket = await callOtto();
  const room = fakes.room()!;
  ready(socket);
  await screen.findByText("You're on the call with Otto");
  return { socket, room };
}

async function liveCall() {
  return (await liveRoomCall()).socket;
}

function endCall(socket: FakeWebSocket, reason: string) {
  act(() => {
    socket.relay({ type: "ended", reason, elapsed_s: 92.4 });
    socket.close(1000);
  });
}

// The relay's answer to a start: the socket opens, and the relay refuses.
async function refusedStart(code: number, error?: Record<string, unknown>) {
  const socket = await callOtto();
  act(() => {
    if (error) socket.relay({ type: "error", ...error });
    socket.drop(code);
  });
  return socket;
}

// A finish of the call that answers only when the test lets it.
function holdFinish() {
  let answer: () => void = () => undefined;
  const answered = new Promise<void>((resolve) => {
    answer = resolve;
  });
  server.use(
    http.post(`${VOICE}/:callId/finish`, async ({ params }) => {
      finishes.push(String(params.callId));
      await answered;
      return HttpResponse.json(ACCEPTED);
    }),
  );
  return answer;
}

// Moves the clock on by *ms*, for a check that something does not happen:
// the test runs with fake timers that otherwise keep pace with real time.
async function passTime(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
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
  mintAnswer = STARTED;
  callStates = {};
  typedMessages = [KICKOFF_ROW, GREETING_ROW];
  turnAnswer = null;
  mints = 0;
  mintBodies = [];
  latestReads = 0;
  statusReads = [];
  finishes = [];
  ends = [];
  typedStartCount = 0;
  typedReads = 0;
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
  useCopilotUIStore.getState().setInitialPrompt(null);
  localStorage.clear();
  sessionStorage.clear();
  sdkFails = false;
  vi.mocked(reportCallStart).mockClear();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
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

  it("falls back to the choice when the latest call's read does not answer in time", async () => {
    server.use(http.get(VOICE, () => new Promise<never>(() => undefined)));

    render(<OnboardingPage />);

    expect(
      await screen.findByRole("button", { name: "Talk to Otto" }),
    ).toBeDefined();
  });

  it("starts no call on a browser that cannot join a call's room", async () => {
    fakes.unsupportBrowser();

    render(<OnboardingPage />);
    await talkToOtto();

    expect(
      await screen.findByText(
        "This browser can't make calls. You can type instead, or call from another browser.",
      ),
    ).toBeDefined();
    expect(mints).toBe(0);
    expect(tracked("voice_interview_start_failed")).toEqual([
      { stage: "browser", why: "unsupported" },
    ]);
    const microphone = await fakes.getUserMedia.mock.results[0].value;
    expect(microphone.getTracks()[0].stop).toHaveBeenCalled();
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Type instead" }));
    expect(await screen.findByText(GREETING)).toBeDefined();
  });

  it("starts no call when the room's SDK does not load", async () => {
    sdkFails = true;

    render(<OnboardingPage />);
    await talkToOtto();

    expect(
      await screen.findByText(
        "We couldn't start the call. Try again, or type instead.",
      ),
    ).toBeDefined();
    expect(mints).toBe(0);
    expect(fakes.rooms()).toEqual([]);
    expect(tracked("voice_interview_start_failed")).toEqual([
      { stage: "browser", why: "sdk" },
    ]);
  });

  it("lets go of the call's audio when the start is refused", async () => {
    refuseCallStartOnce({
      status: 429,
      body: { detail: { message: "Calls used.", retry_after_seconds: 600 } },
    });

    render(<OnboardingPage />);
    await talkToOtto();
    await screen.findByText(/You've made a few calls already/);

    // The context the click made, beside the aura's own.
    expect(fakes.contexts()[0].close).toHaveBeenCalled();
    const microphone = await fakes.getUserMedia.mock.results[0].value;
    expect(microphone.getTracks()[0].stop).toHaveBeenCalled();
    expect(fakes.rooms()).toEqual([]);
  });

  it("shows the live call the relay names when it refuses a start for it", async () => {
    render(<OnboardingPage />);

    await refusedStart(1008, {
      error: "call_in_progress",
      call_id: OTHER_CALL_ID,
    });

    expect(
      await screen.findByRole("button", { name: "End that call" }),
    ).toBeDefined();
  });

  it("tries the start again with the same token when the relay asks, then says calls are unavailable", async () => {
    render(<OnboardingPage />);
    let socket = await refusedStart(1013);
    for (let retry = 0; retry < 3; retry += 1) {
      socket = await relaySocketAfter(socket);
      act(() => socket.accept());
      act(() => socket.drop(1013));
    }

    expect(
      await screen.findByText(
        "Otto can't take calls right now. You can type instead, or try again in a moment.",
      ),
    ).toBeDefined();
    expect(fakes.sockets).toHaveLength(4);
    expect(fakes.sockets.map((each) => each.messages()[0])).toEqual(
      Array.from({ length: 4 }, () => ({ type: "start", token: TOKEN })),
    );
    expect(mints).toBe(1);
  });

  it("says the start failed when the relay refuses it outright", async () => {
    render(<OnboardingPage />);

    await refusedStart(1008, { error: "token_invalid" });

    expect(
      await screen.findByText(
        "We couldn't start the call. Try again, or type instead.",
      ),
    ).toBeDefined();
  });

  it("treats a refusal that names no call id as a start that failed", async () => {
    refuseCallStartOnce({
      status: 409,
      body: { detail: { code: "call_in_progress", call_id: "../../x/end" } },
    });

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Talk to Otto" }));

    expect(
      await screen.findByText(
        "We couldn't start the call. Try again, or type instead.",
      ),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "End that call" })).toBeNull();
    expect(requests.filter((path) => path.includes(".."))).toEqual([]);
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
    expect(socket.messages()[0]).toEqual({ type: "start", token: TOKEN });
  });
});

describe("onboarding interview by voice: the call", () => {
  it("starts the clock and the turns when the call goes live", async () => {
    render(<OnboardingPage />);
    const socket = await callOtto();
    expect(socket.url).toBe(RELAY_URL);
    expect(socket.messages()[0]).toEqual({ type: "start", token: TOKEN });

    ready(socket);

    expect(await screen.findByText("5:00 left")).toBeDefined();
    expect(screen.getByText("16 turns left")).toBeDefined();
    // How long each step of the start took, from the microphone on, and
    // what the browser applied to the microphone.
    expect(tracked("interview_started")).toEqual([
      {
        mode: "voice",
        mint_s: expect.any(Number),
        room_s: expect.any(Number),
        ready_s: expect.any(Number),
        echo_cancellation: true,
        noise_suppression: true,
        auto_gain_control: false,
      },
    ]);
  });

  it("tells the person Otto can hear himself once the relay stops listening over him, for the rest of the call", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();
    expect(screen.queryByText(HEARS_HIMSELF)).toBeNull();

    act(() => socket.relay({ type: "duplex", mode: "half", by: "probe" }));

    const line = await screen.findByText(HEARS_HIMSELF);
    expect(line.getAttribute("role")).toBe("status");
    expect(tracked("voice_interview_echo")).toEqual([{ by: "probe" }]);
    act(() => socket.relay({ type: "turn_complete", turn: 0, turns_left: 15 }));
    expect(await screen.findByText("15 turns left")).toBeDefined();
    expect(screen.getByText(HEARS_HIMSELF)).toBeDefined();

    endCall(socket, "finished");
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Hang up" })).toBeNull(),
    );
    expect(screen.queryByText(HEARS_HIMSELF)).toBeNull();
  });

  it("says nothing of Otto hearing himself on a call the relay keeps listening on", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();

    act(() => {
      socket.relay({ type: "agent_text", text: "Hi, I'm Otto." });
      socket.relay({ type: "turn_complete", turn: 0, turns_left: 15 });
      socket.relay({ type: "user_text", text: "I run a bakery." });
    });

    expect(await screen.findByText("I run a bakery.")).toBeDefined();
    expect(screen.queryByText(HEARS_HIMSELF)).toBeNull();
    expect(tracked("voice_interview_echo")).toEqual([]);
  });

  it("reports how long the call took to reach Otto's first sound, once", async () => {
    render(<OnboardingPage />);
    const { room } = await liveRoomCall();
    act(() => {
      room.ottoJoins();
    });
    expect(reportCallStart).not.toHaveBeenCalled();

    act(() => fakes.ottoSounds());

    await waitFor(() => expect(reportCallStart).toHaveBeenCalledTimes(1));
    const [callId, marks] = vi.mocked(reportCallStart).mock.calls[0];
    expect(callId).toBe(CALL_ID);
    const steps = [
      marks.granted,
      marks.minted,
      marks.joined,
      marks.ready,
      marks.firstSound,
    ];
    // Every step reached, each no sooner than the one before it.
    expect(steps.every((step) => typeof step === "number")).toBe(true);
    const times = steps.map(Number);
    expect([...times].sort((a, b) => a - b)).toEqual(times);
    // Otto's speech arriving again in the room is no second start.
    act(() => {
      room.ottoJoins();
    });
    await act(() => new Promise((resolve) => setTimeout(resolve, 200)));
    expect(reportCallStart).toHaveBeenCalledTimes(1);
  });

  it("reports the start once Otto is heard before the call's ready reaches the page", async () => {
    render(<OnboardingPage />);
    const socket = await callOtto();
    act(() => {
      fakes.room()!.ottoJoins();
      fakes.ottoSounds();
    });
    // Otto's speech is heard while the page still waits for the ready.
    await act(() => new Promise((resolve) => setTimeout(resolve, 200)));
    expect(reportCallStart).not.toHaveBeenCalled();

    ready(socket);

    await waitFor(() => expect(reportCallStart).toHaveBeenCalledTimes(1));
    const [callId, marks] = vi.mocked(reportCallStart).mock.calls[0];
    expect(callId).toBe(CALL_ID);
    expect(marks.firstSound).not.toBeNull();
    expect(Number(marks.firstSound)).toBeLessThanOrEqual(Number(marks.ready));
  });

  it("reports how Otto's speech fared in the room when the call ends", async () => {
    render(<OnboardingPage />);
    const { socket, room } = await liveRoomCall();
    act(() => {
      room.ottoJoins();
      fakes.ottoStats(OTTO_STATS);
    });
    await waitFor(() => expect(fakes.ottoStatsReads()).toBeGreaterThan(0));

    endCall(socket, "finished");

    expect(tracked("interview_ended")).toEqual([
      {
        mode: "voice",
        reason: "finished",
        elapsed_s: 92.4,
        turns: 0,
        ...OTTO_HEARD,
      },
    ]);
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
    // The control socket came back; the call's room never went.
    expect(tracked("voice_interview_resumed")).toEqual([{ gap_s: 1.2 }]);
    expect(tracked("voice_interview_reconnected")).toEqual([]);
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
    latestCall = liveRoomRecord();
    callStates[CALL_ID] = [latestCall];

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Rejoin" }));

    const socket = await relaySocketAfter();
    // The socket the server named for the call, read at the click.
    expect(socket.url).toBe(RELAY_URL);
    act(() => socket.accept());
    expect(socket.messages()[0]).toEqual({ type: "resume", ticket: TICKET });
    act(() => socket.relay({ type: "resumed", gap_s: 4.1 }));

    expect(await screen.findByText("Reconnected")).toBeDefined();
    expect(screen.getByText("12 turns left")).toBeDefined();
    expect(mints).toBe(0);
  });

  it("counts the turns heard before a rejoin when the call ends", async () => {
    keepTicket();
    latestCall = liveRoomRecord();
    callStates[CALL_ID] = [latestCall];

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Rejoin" }));
    const socket = await relaySocketAfter();
    act(() => socket.accept());
    act(() => socket.relay({ type: "resumed", gap_s: 4.1 }));
    callStates[CALL_ID] = [callRecord({ interview: "taken" })];
    endCall(socket, "finished");

    await screen.findByRole("button", { name: "Meet your team" });
    // The ticket kept 12 of 16 turns left: four were heard.
    expect(tracked("interview_ended")).toEqual([
      {
        mode: "voice",
        reason: "finished",
        elapsed_s: 92.4,
        turns: 4,
        ...NOT_LISTENED,
      },
    ]);
  });

  it("reports how Otto's speech fared in the room of a rejoined call", async () => {
    keepTicket();
    latestCall = liveRoomRecord();
    callStates[CALL_ID] = [latestCall];

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Rejoin" }));
    const socket = await relaySocketAfter();
    act(() => socket.accept());
    act(() => socket.relay({ type: "resumed", gap_s: 4.1 }));
    await waitFor(() => expect(fakes.room()?.url).toBe(ROOM_URL));
    act(() => {
      fakes.room()!.ottoJoins();
      fakes.ottoStats(OTTO_STATS);
    });
    await waitFor(() => expect(fakes.ottoStatsReads()).toBeGreaterThan(0));
    callStates[CALL_ID] = [callRecord({ interview: "taken" })];
    endCall(socket, "finished");

    await screen.findByRole("button", { name: "Meet your team" });
    expect(tracked("interview_ended")).toEqual([
      expect.objectContaining({ reason: "finished", ...OTTO_HEARD }),
    ]);
  });

  it("offers no Type instead while a dropped call re-attaches", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();

    act(() => socket.drop(1006));

    expect(await screen.findByText("Reconnecting…")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Type instead" })).toBeNull();
  });

  it("keeps Skip away while the call is live and while its words are saved", async () => {
    callStates[CALL_ID] = [
      callRecord({ status: "transcribing", hand_off: null }),
    ];
    render(<OnboardingPage />);
    const socket = await liveCall();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();

    endCall(socket, "finished");

    expect(await screen.findByText("Wrapping up…")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
  });

  it("ends a live call through the REST API when the person types instead, then opens the typed interview", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Type instead" }));

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(ends).toEqual([CALL_ID]);
    // The page told the relay nothing: the end route ends the call.
    expect(socket.messages().filter((m) => m.type === "hangup")).toEqual([]);
    expect(tracked("interview_ended")).toEqual([
      {
        mode: "voice",
        reason: "type_instead",
        elapsed_s: expect.any(Number),
        turns: 0,
        ...NOT_LISTENED,
      },
    ]);
  });

  it("loses a dropped call once the relay's window passes, then reads its words", async () => {
    callStates[CALL_ID] = [callRecord({ end_reason: "client_gone" })];
    render(<OnboardingPage />);
    const socket = await liveCall();

    // The network is gone: the re-attach never opens.
    act(() => socket.drop(1006));
    expect(await screen.findByText("Reconnecting…")).toBeDefined();

    expect(
      await screen.findByText(
        "The connection dropped.",
        {},
        { timeout: 6_000 },
      ),
    ).toBeDefined();
    expect(screen.getByText("What you said so far is saved.")).toBeDefined();
    expect(statusReads).toContain(CALL_ID);
  });

  it("shows the ending of a call that ended before the rejoin, and opens no socket", async () => {
    keepTicket();
    latestCall = callRecord({
      status: "live",
      end_reason: null,
      hand_off: null,
    });
    callStates[CALL_ID] = [callRecord({ interview: "taken" })];

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Rejoin" }));

    expect(
      await screen.findByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(fakes.sockets).toEqual([]);
  });

  it("lets the call go on in the tab that rejoined it, and takes nothing back", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    render(<OnboardingPage />);
    const socket = await liveCall();
    const microphone = await fakes.getUserMedia.mock.results[0].value;

    // The relay closes this socket because another tab rejoined the call.
    act(() => socket.drop(4000));

    expect(
      await screen.findByText(
        "Your call with Otto went on in another tab or on another device.",
      ),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "End that call" })).toBeDefined();
    // Past the whole re-attach window (2 s here): no socket tries again.
    await passTime(3_000);
    expect(fakes.sockets).toEqual([socket]);
    expect(microphone.getTracks()[0].stop).toHaveBeenCalled();
  });

  it("stops the microphone as soon as the person hangs up", async () => {
    render(<OnboardingPage />);
    await liveCall();
    const microphone = await fakes.getUserMedia.mock.results[0].value;

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Hang up" }));

    expect(microphone.getTracks()[0].stop).toHaveBeenCalled();
  });

  it("puts no audio on the call's socket, and plays none it is sent", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();

    act(() => {
      socket.relayBinary();
      socket.relay({ type: "interrupted" });
      socket.relay({ type: "agent_text", text: "Still here." });
    });

    expect(await screen.findByText("Still here.")).toBeDefined();
    expect(socket.sent.every((data) => typeof data === "string")).toBe(true);
  });

  it("says Hang up again on the socket a re-attach brings", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Hang up" }));

    act(() => socket.drop(1006));
    const again = await relaySocketAfter(socket);
    act(() => again.accept());
    act(() => again.relay({ type: "resumed", gap_s: 0.8 }));

    expect(again.messages()).toEqual([
      { type: "resume", ticket: TICKET },
      { type: "hangup" },
    ]);
  });

  it("ends the call through the REST API when the relay does not answer Hang up", async () => {
    render(<OnboardingPage />);
    await liveCall();

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Hang up" }));
    expect(ends).toEqual([]);

    await waitFor(() => expect(ends).toEqual([CALL_ID]));
  });

  it("keeps the captions out of session replays", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();

    act(() => socket.relay({ type: "user_text", text: "I run a bakery." }));

    const captions = await screen.findByRole("log", { name: "Call captions" });
    expect(captions.classList.contains("ph-no-capture")).toBe(true);
  });

  it("carries on with the call past frames the relay never sends", async () => {
    render(<OnboardingPage />);
    const socket = await liveCall();

    act(() => {
      socket.relay({ type: "agent_text" });
      socket.relay({ type: "turn_complete", turn: 0, turns_left: "many" });
      socket.relay({ type: "ready", call_id: CALL_ID });
      socket.relay({ type: "leg", leg: "websocket" });
      socket.relay({ type: "agent_text", text: "Still here." });
    });

    expect(await screen.findByText("Still here.")).toBeDefined();
    expect(screen.getByText("16 turns left")).toBeDefined();
    expect(screen.getByText("You're on the call with Otto")).toBeDefined();
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
      {
        mode: "voice",
        reason: "finished",
        elapsed_s: 92.4,
        turns: 0,
        ...NOT_LISTENED,
      },
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

  it("offers another call or typing, never Try again, for a call whose words were not kept", async () => {
    callStates[CALL_ID] = [
      callRecord({
        status: "failed",
        hand_off: {
          status: "failed",
          input_mode: "typed",
          error_code: "some_new_failure",
        },
      }),
    ];
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "finished");

    expect(await screen.findByText(SAVE_FAILED)).toBeDefined();
    expect(screen.getByRole("button", { name: "Call again" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Type instead" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
    expect(screen.queryByText(/What you said is kept/)).toBeNull();
  });

  it.each([
    ["quality_check_unavailable", "Try again"],
    ["some_new_failure", "Call again"],
    ["no_usable_speech", "Call again"],
  ])(
    "never says Otto had everything it needed when a finished call's hand-off failed (%s)",
    async (code, wayOn) => {
      callStates[CALL_ID] = [
        callRecord({
          status: "failed",
          end_reason: "finished",
          hand_off: { status: "failed", input_mode: "typed", error_code: code },
        }),
      ];
      render(<OnboardingPage />);
      const socket = await liveCall();

      endCall(socket, "finished");

      expect(await screen.findByRole("button", { name: wayOn })).toBeDefined();
      expect(screen.getByText("Otto wrapped up the call.")).toBeDefined();
      expect(screen.queryByText("That's everything Otto needed.")).toBeNull();
    },
  );

  it("says it did not hear the person, with another call or typing, after a call where nothing was heard", async () => {
    callStates[CALL_ID] = [
      callRecord({
        status: "failed",
        end_reason: "user_hangup",
        hand_off: {
          status: "failed",
          input_mode: "typed",
          error_code: "no_usable_speech",
        },
      }),
    ];
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "user_hangup");

    expect(await screen.findByText(DIDNT_HEAR_YOU)).toBeDefined();
    expect(screen.getByRole("button", { name: "Call again" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Type instead" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
    expect(screen.queryByText(/We heard you/)).toBeNull();
  });

  it.each([["transcription_unavailable"], ["recording_unavailable"]])(
    "says something went wrong saving the call, never that the person was not heard, after %s",
    async (code) => {
      callStates[CALL_ID] = [
        callRecord({
          status: "failed",
          end_reason: "user_hangup",
          hand_off: { status: "failed", input_mode: "typed", error_code: code },
        }),
      ];
      render(<OnboardingPage />);
      const socket = await liveCall();

      endCall(socket, "user_hangup");

      expect(await screen.findByText(SAVE_FAILED)).toBeDefined();
      expect(screen.getByRole("button", { name: "Call again" })).toBeDefined();
      expect(
        screen.getByRole("button", { name: "Type instead" }),
      ).toBeDefined();
      expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
      expect(
        screen.queryByText(/didn't hear|heard you|nothing was heard/i),
      ).toBeNull();
    },
  );

  it("says it did not hear the person when a call's finish answers that nothing was heard", async () => {
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
    server.use(
      getFinishOnboardingVoiceCallMockHandler200(({ params }) => {
        finishes.push(String(params.callId));
        return {
          status: "failed",
          input_mode: "typed",
          error_code: "no_usable_speech",
        };
      }),
    );
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "finished");
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Try again" }));

    expect(await screen.findByText(DIDNT_HEAR_YOU)).toBeDefined();
    expect(finishes).toEqual([CALL_ID]);
    expect(screen.queryByTestId("step-plan")).toBeNull();
  });

  it("keeps the call's ending and says the finish failed when the finish route errs", async () => {
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
    server.use(
      http.post(`${VOICE}/:callId/finish`, ({ params }) => {
        finishes.push(String(params.callId));
        return HttpResponse.json({ detail: "Internal error" }, { status: 500 });
      }),
    );
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "finished");
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Try again" }));

    expect(
      await screen.findByText("That didn't go through. Try again."),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Try again" })).toHaveProperty(
      "disabled",
      false,
    );
    expect(finishes).toEqual([CALL_ID]);
  });

  it("keeps Skip and Back away while a call's finish is on its way", async () => {
    callStates[CALL_ID] = [callRecord({ end_reason: "vendor_error" })];
    const answerFinish = holdFinish();
    render(<OnboardingPage />);
    const socket = await liveCall();
    endCall(socket, "vendor_error");

    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Meet your team" }));
    await waitFor(() => expect(finishes).toEqual([CALL_ID]));

    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Back" })).toBeNull();
    answerFinish();
    await expectPlanStep();
  });

  it("never moves on when the step goes away while a call's finish is on its way", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    callStates[CALL_ID] = [callRecord({ end_reason: "vendor_error" })];
    const answerFinish = holdFinish();
    render(<OnboardingPage />);
    const socket = await liveCall();
    endCall(socket, "vendor_error");
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Meet your team" }));
    await waitFor(() => expect(finishes).toEqual([CALL_ID]));
    const stepBefore = useOnboardingWizardStore.getState().currentStep;

    cleanup();
    answerFinish();
    await passTime(1_000);

    expect(useOnboardingWizardStore.getState().currentStep).toBe(stepBefore);
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBeNull();
  });

  it("reports an ending once per call in a tab", async () => {
    callStates[CALL_ID] = [callRecord({ interview: "taken" })];
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "finished");
    await screen.findByRole("button", { name: "Meet your team" });

    expect(
      vi
        .mocked(capturePostHogEvent)
        .mock.calls.filter(([name]) => name === "ending_shown"),
    ).toEqual([
      [
        "ending_shown",
        { mode: "voice", ending: "team" },
        { oncePerTabKey: `voice_ending_shown_${CALL_ID}_team` },
      ],
    ]);
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

  it("opens the typed interview when the call's finish answers that the person's typing superseded it", async () => {
    callStates[CALL_ID] = [callRecord({ end_reason: "vendor_error" })];
    server.use(
      getFinishOnboardingVoiceCallMockHandler200(({ params }) => {
        finishes.push(String(params.callId));
        // The person typed into their interview after the call ended.
        callStates[CALL_ID] = [
          callRecord({ end_reason: "vendor_error", interview: "superseded" }),
        ];
        return SUPERSEDED_TAKE;
      }),
    );
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "vendor_error");
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Meet your team" }));

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(finishes).toEqual([CALL_ID]);
    expect(screen.queryByTestId("step-plan")).toBeNull();
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBeNull();
  });

  it("keeps Saving what you said while a call's finish waits on a later call, then moves on", async () => {
    callStates[CALL_ID] = [callRecord({ end_reason: "vendor_error" })];
    let laterCallSettled = false;
    server.use(
      http.post(`${VOICE}/:callId/finish`, ({ params }) => {
        finishes.push(String(params.callId));
        // A later call of the person's is still unsettled.
        if (!laterCallSettled) {
          return HttpResponse.json(
            { detail: { code: "call_not_ready" } },
            { status: 409 },
          );
        }
        return HttpResponse.json(ACCEPTED);
      }),
    );
    render(<OnboardingPage />);
    const socket = await liveCall();

    endCall(socket, "vendor_error");
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Meet your team" }));

    expect(await screen.findByText("Saving what you said…")).toBeDefined();
    await waitFor(() => expect(finishes.length).toBeGreaterThanOrEqual(3));
    expect(screen.queryByText("That didn't go through. Try again.")).toBeNull();
    expect(screen.queryByRole("button", { name: "Meet your team" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();

    laterCallSettled = true;

    await expectPlanStep();
    expect(new Set(finishes)).toEqual(new Set([CALL_ID]));
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
  });

  it("offers Type instead past the wait while a call's finish waits, and stops finishing once the person types", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    callStates[CALL_ID] = [callRecord({ end_reason: "vendor_error" })];
    server.use(
      http.post(`${VOICE}/:callId/finish`, ({ params }) => {
        finishes.push(String(params.callId));
        return HttpResponse.json(
          { detail: { code: "call_not_ready" } },
          { status: 409 },
        );
      }),
    );
    render(<OnboardingPage />);
    const socket = await liveCall();
    const user = userEvent.setup();

    endCall(socket, "vendor_error");
    await user.click(
      await screen.findByRole("button", { name: "Meet your team" }),
    );
    expect(await screen.findByText("Saving what you said…")).toBeDefined();
    await user.click(
      await screen.findByRole("button", { name: "Type instead" }),
    );

    expect(await screen.findByText(GREETING)).toBeDefined();
    // A finish already on its way lands first; after it, none is asked
    // again, well past the wait's slower reads (40 ms apart here).
    await passTime(100);
    const asked = finishes.length;
    await passTime(1_000);
    expect(finishes.length).toBe(asked);
    expect(screen.queryByTestId("step-plan")).toBeNull();
  });

  it("moves on as path A when the call's finish answers that another take beat it", async () => {
    callStates[CALL_ID] = [callRecord({ end_reason: "vendor_error" })];
    server.use(
      getFinishOnboardingVoiceCallMockHandler200(({ params }) => {
        finishes.push(String(params.callId));
        return SUPERSEDED_TAKE;
      }),
    );
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

  it("reports how a lost call ended when the person types before its words have an answer", async () => {
    callStates[CALL_ID] = [
      callRecord({ status: "live", end_reason: null, hand_off: null }),
      callRecord({
        status: "transcribing",
        end_reason: "client_gone",
        hand_off: null,
      }),
    ];
    render(<OnboardingPage />);
    const socket = await liveCall();

    act(() => socket.drop(1006));
    const again = await relaySocketAfter(socket);
    act(() => again.drop(4404));
    expect(
      await screen.findByText("We lost the call. Saving what you said…"),
    ).toBeDefined();
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Type instead" }));

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(tracked("interview_ended")).toEqual([
      {
        mode: "voice",
        reason: "client_gone",
        elapsed_s: expect.any(Number),
        turns: 0,
        ...NOT_LISTENED,
      },
    ]);
  });

  it("offers typing, another call or skipping when the call's words said too little", async () => {
    callStates[CALL_ID] = [
      callRecord({
        status: "failed",
        hand_off: {
          status: "failed",
          input_mode: "typed",
          error_code: "insufficient_content",
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
    // A call keeps its own not-enough screen; the brain dump's asks its
    // question instead.
    expect(
      screen.queryByRole("heading", {
        name: "What do you want off your plate?",
      }),
    ).toBeNull();

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Call again" }));

    await relaySocketAfter(socket);
    expect(mints).toBe(2);
  });

  it.each(["transcribing", "handing_off"] as const)(
    "keeps Wrapping up beside Type instead past the wait while the call is %s, and stops reading the call once the person types",
    async (status) => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      callStates[CALL_ID] = [callRecord({ status, hand_off: null })];
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
      // Well past the wait's slower reads (40 ms apart here).
      await passTime(1_000);

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

  it("takes a turn refused for a live call off the chat and back into the composer", async () => {
    turnAnswer = {
      status: 409,
      body: { detail: { code: "call_in_progress", call_id: OTHER_CALL_ID } },
    };
    callStates[OTHER_CALL_ID] = [
      callRecord({ call_id: OTHER_CALL_ID, status: "live", end_reason: null }),
    ];
    render(<OnboardingPage />);
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Type instead" }),
    );
    const input = await screen.findByPlaceholderText("Type your answer…");
    await waitFor(() => expect(input).toHaveProperty("disabled", false));

    await user.type(input, "One more thing.");
    await user.keyboard("{Enter}");

    const endThatCall = await screen.findByRole("button", {
      name: "End that call",
    });
    expect(useCopilotUIStore.getState().initialPrompt).toBe("One more thing.");

    turnAnswer = null;
    await user.click(endThatCall);

    const again = await screen.findByPlaceholderText("Type your answer…");
    await waitFor(() =>
      expect(again).toHaveProperty("value", "One more thing."),
    );
    // The refused message is not on the chat, and counts as no answer.
    expect(
      screen.queryByText("One more thing.", {
        ignore: "textarea, script, style",
      }),
    ).toBeNull();
    expect(screen.queryByRole("button", { name: "I'm done" })).toBeNull();
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

  it("shows the call's ending when the typed finish is refused for a call that took the interview", async () => {
    typedMessages = [KICKOFF_ROW, GREETING_ROW, TYPED_ANSWER_ROW];
    callStates[CALL_ID] = [callRecord({ interview: "taken" })];
    refuseTypedFinish({
      status: 409,
      body: {
        detail: { code: "onboarding_interview_closed", call_id: CALL_ID },
      },
    });
    render(<OnboardingPage />);
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Type instead" }),
    );
    await user.click(await screen.findByRole("button", { name: "I'm done" }));

    expect(
      await screen.findByText("That's everything Otto needed."),
    ).toBeDefined();
    await user.click(screen.getByRole("button", { name: "Meet your team" }));

    await expectPlanStep();
    expect(finishes).toEqual([]);
    expect(sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
  });

  it("offers End that call when the typed finish is refused for a live call", async () => {
    typedMessages = [KICKOFF_ROW, GREETING_ROW, TYPED_ANSWER_ROW];
    callStates[OTHER_CALL_ID] = [
      callRecord({ call_id: OTHER_CALL_ID, status: "live", end_reason: null }),
    ];
    refuseTypedFinish({
      status: 409,
      body: { detail: { code: "call_in_progress", call_id: OTHER_CALL_ID } },
    });
    render(<OnboardingPage />);
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Type instead" }),
    );
    await user.click(await screen.findByRole("button", { name: "I'm done" }));

    expect(
      await screen.findByText(
        "Your call with Otto is still going in another tab or on another device.",
      ),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "End that call" })).toBeDefined();
    expect(screen.queryByText("That didn't go through. Try again.")).toBeNull();
    expect(screen.queryByTestId("step-plan")).toBeNull();
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

  it("shows the earlier call a reload answers, not the call this tab made last", async () => {
    // This tab's last call settled without words, so the server's read
    // looks through it to the call before.
    keepTicket(OTHER_CALL_ID);
    latestCall = callRecord({ end_reason: "client_gone" });

    render(<OnboardingPage />);

    expect(await screen.findByText("The connection dropped.")).toBeDefined();
    expect(screen.getByText("What you said so far is saved.")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Rejoin" })).toBeNull();
    expect(statusReads).not.toContain(OTHER_CALL_ID);
    expect(fakes.sockets).toHaveLength(0);
  });

  it("opens a finished typed interview as its transcript in a new tab, with no choice and no Skip", async () => {
    typedMessages = [
      KICKOFF_ROW,
      GREETING_ROW,
      TYPED_ANSWER_ROW,
      TYPED_GOODBYE_ROW,
    ];
    let answerTypedRead: () => void = () => undefined;
    const typedReadAnswered = new Promise<void>((resolve) => {
      answerTypedRead = resolve;
    });
    server.use(
      getGetOnboardingInterviewMockHandler200(async () => {
        typedReads += 1;
        await typedReadAnswered;
        return {
          session_id: SESSION_ID,
          stage: "handed_off",
          is_over: true,
          voice_available: true,
        };
      }),
      getStartOnboardingInterviewMockHandler200(() => {
        typedStartCount += 1;
        return { session_id: SESSION_ID, turn_id: null, stage: "handed_off" };
      }),
    );

    render(<OnboardingPage />);
    await waitFor(() => expect(typedReads).toBe(1));
    // The latest call was read first; while the typed interview is read,
    // neither the choice nor its Skip shows.
    expect(latestReads).toBe(1);
    expect(screen.queryByRole("button", { name: "Talk to Otto" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();

    answerTypedRead();

    expect(
      await screen.findByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
    expect(await screen.findByText(TYPED_GOODBYE)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Talk to Otto" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
    expect(screen.queryByPlaceholderText("Type your answer…")).toBeNull();
  });

  it.each([
    ["has none", { status: 200, body: NO_INTERVIEW }],
    ["cannot be read", { status: 503, body: { detail: "Unavailable" } }],
    [
      "is under way",
      {
        status: 200,
        body: { ...NO_INTERVIEW, session_id: SESSION_ID },
      },
    ],
  ])(
    "shows the choice with Skip when the person's typed interview %s",
    async (_case, answer: Answer) => {
      server.use(
        http.get("/api/proxy/api/onboarding/brain-dump/interview", () => {
          typedReads += 1;
          return HttpResponse.json(answer.body ?? {}, {
            status: answer.status,
          });
        }),
      );

      render(<OnboardingPage />);

      expect(
        await screen.findByRole("button", { name: "Talk to Otto" }),
      ).toBeDefined();
      expect(
        screen.getByRole("button", { name: "Skip for now" }),
      ).toBeDefined();
      expect(typedReads).toBe(1);
      expect(typedStartCount).toBe(0);
    },
  );

  it("opens the typed interview, with no choice and no word about the microphone, where a call cannot be offered", async () => {
    fakes.denyMicrophonePermission();
    server.use(
      http.get("/api/proxy/api/onboarding/brain-dump/interview", () => {
        typedReads += 1;
        return HttpResponse.json({ ...NO_INTERVIEW, voice_available: false });
      }),
    );

    render(<OnboardingPage />);

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Talk to Otto" })).toBeNull();
    expect(
      screen.queryByText(
        "Your browser blocked the microphone, so write here instead.",
      ),
    ).toBeNull();
    expect(latestReads).toBe(1);
    expect(mints).toBe(0);
    // Not the person's choice: a later visit reads again.
    expect(
      sessionStorage.getItem(`autogpt:onboarding-interview-typing:${USER_ID}`),
    ).toBeNull();
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

describe("onboarding interview by voice: the call's room", () => {
  const AUDIO_FAILED =
    "We couldn't connect the call's audio. Try again, or type instead.";

  it("joins the call's room before it starts the call, and sends no audio on the socket", async () => {
    render(<OnboardingPage />);
    const { socket, room } = await liveRoomCall();

    // The start asks for nothing: every call takes its room.
    expect(mintBodies).toEqual([""]);
    expect(room.url).toBe(ROOM_URL);
    expect(room.token).toBe(ROOM_TOKEN);
    // The microphone the call opened, published once, as a microphone.
    expect(room.published).toHaveLength(1);
    expect(room.publishOptions).toEqual([{ source: "microphone" }]);
    expect(fakes.getUserMedia).toHaveBeenCalledTimes(1);
    expect(socket.messages()[0]).toEqual({ type: "start", token: TOKEN });
    expect(socket.sent.every((data) => typeof data === "string")).toBe(true);
  });

  it("plays Otto's speech from the room through the player the click started", async () => {
    render(<OnboardingPage />);
    const { room } = await liveRoomCall();

    let otto: FakeRemoteTrack | undefined;
    act(() => {
      otto = room.ottoJoins();
    });

    expect(otto?.attachedTo).toHaveLength(1);
    expect(otto?.attachedTo[0]).toBeInstanceOf(HTMLAudioElement);
  });

  it.each([
    [
      "refuses",
      { name: "ConnectionError", reasonName: "NotAllowed", status: 403 },
      "refused",
    ],
    ["fails", new Error("the network is gone"), "failed"],
  ])(
    "says the call's audio could not connect, and ends the call, when LiveKit %s the join",
    async (_case, error, why) => {
      fakes.refuseRoomJoins(error);
      render(<OnboardingPage />);
      await talkToOtto();

      expect(await screen.findByText(AUDIO_FAILED)).toBeDefined();
      // The call never started on the relay; its room closes now.
      expect(fakes.sockets).toEqual([]);
      expect(fakes.room()?.isLeft).toBe(true);
      await waitFor(() => expect(ends).toEqual([CALL_ID]));
      expect(tracked("voice_interview_start_failed")).toEqual([
        { stage: "room", why },
      ]);
      expect(tracked("interview_started")).toEqual([]);
      expect(screen.queryByText(DIDNT_HEAR_YOU)).toBeNull();
    },
  );

  it("says the call's audio could not connect when the join takes too long", async () => {
    fakes.hangRoomJoins();
    render(<OnboardingPage />);
    await talkToOtto();

    expect(await screen.findByText(AUDIO_FAILED)).toBeDefined();
    // The join that never answered is let go.
    expect(fakes.room()?.isLeft).toBe(true);
    expect(fakes.sockets).toEqual([]);
    await waitFor(() => expect(ends).toEqual([CALL_ID]));
    expect(tracked("voice_interview_start_failed")).toEqual([
      { stage: "room", why: "timeout" },
    ]);
  });

  it("starts a fresh call from Talk to Otto after the audio could not connect", async () => {
    fakes.refuseRoomJoins(new Error("the network is gone"));
    render(<OnboardingPage />);
    await talkToOtto();
    expect(await screen.findByText(AUDIO_FAILED)).toBeDefined();

    fakes.joinRooms();
    const socket = await callOtto();
    ready(socket);

    expect(
      await screen.findByText("You're on the call with Otto"),
    ).toBeDefined();
    expect(mints).toBe(2);
    expect(screen.queryByText(AUDIO_FAILED)).toBeNull();
  });

  it("types instead after the audio could not connect", async () => {
    fakes.refuseRoomJoins(new Error("the network is gone"));
    render(<OnboardingPage />);
    await talkToOtto();
    expect(await screen.findByText(AUDIO_FAILED)).toBeDefined();

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Type instead" }));

    expect(await screen.findByText(GREETING)).toBeDefined();
  });

  it.each([["room_error"], ["vendor_error"]])(
    "says the call's audio could not connect when the relay ends it %s before it went live",
    async (reason) => {
      callStates[CALL_ID] = [callRecord({ end_reason: reason })];
      render(<OnboardingPage />);
      const socket = await callOtto();

      act(() => {
        socket.relay({ type: "ended", reason, elapsed_s: 0 });
        socket.close(1000);
      });

      expect(await screen.findByText(AUDIO_FAILED)).toBeDefined();
      expect(fakes.room()?.isLeft).toBe(true);
      // Never the ending of a call that ran, nor a wait for its words.
      expect(screen.queryByText(DIDNT_HEAR_YOU)).toBeNull();
      expect(screen.queryByText("Wrapping up…")).toBeNull();
      expect(statusReads).toEqual([]);
      expect(tracked("voice_interview_start_failed")).toEqual([
        { stage: "relay", why: reason },
      ]);
      expect(tracked("interview_ended")).toEqual([]);
    },
  );

  it("leaves the room a join lets the browser into after the person moved on to typing", async () => {
    const letIn = fakes.holdRoomJoins();
    render(<OnboardingPage />);
    await talkToOtto();
    await waitFor(() => expect(fakes.room()?.token).toBe(ROOM_TOKEN));

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Type instead" }));
    expect(await screen.findByText(GREETING)).toBeDefined();
    act(() => letIn());

    await waitFor(() => expect(fakes.room()?.isLeft).toBe(true));
    // The call never started on the relay.
    expect(fakes.sockets).toEqual([]);
    expect(tracked("voice_interview_start_failed")).toEqual([]);
  });

  it("shows Reconnecting while LiveKit gets the call's audio back, with captions still arriving", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    render(<OnboardingPage />);
    const { socket, room } = await liveRoomCall();

    act(() => room.reconnecting());
    expect(await screen.findByText("Reconnecting…")).toBeDefined();
    act(() => socket.relay({ type: "agent_text", text: "Still with me?" }));
    expect(await screen.findByText("Still with me?")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Hang up" }).hasAttribute("disabled"),
    ).toBe(false);
    await passTime(2_000);

    act(() => room.reconnected());
    expect(await screen.findByText("Reconnected")).toBeDefined();
    // Reported as LiveKit's reconnect after the two seconds it took, not
    // the call's socket re-attaching.
    expect(tracked("voice_interview_reconnected")).toEqual([
      { gap_s: expect.closeTo(2, 0) },
    ]);
    expect(tracked("voice_interview_resumed")).toEqual([]);
  });

  it("reports no reconnect for a room that never said it was reconnecting", async () => {
    render(<OnboardingPage />);
    const { room } = await liveRoomCall();

    act(() => room.reconnected());

    expect(screen.queryByText("Reconnecting…")).toBeNull();
    expect(tracked("voice_interview_reconnected")).toEqual([]);
  });

  it("stays in the room once the call ends, until Otto's last words have played", async () => {
    callStates[CALL_ID] = [callRecord({ interview: "taken" })];
    render(<OnboardingPage />);
    const { socket, room } = await liveRoomCall();
    act(() => {
      room.ottoJoins();
    });

    endCall(socket, "finished");

    // The microphone left the room with the call; the page stays to hear
    // Otto out.
    expect(room.unpublished).toHaveLength(1);
    expect(room.isLeft).toBe(false);
    act(() => room.ottoLeaves());
    expect(room.isLeft).toBe(true);
    expect(
      await screen.findByRole("button", { name: "Meet your team" }),
    ).toBeDefined();
  });

  it("leaves the room once its wait for Otto's last words runs out", async () => {
    callStates[CALL_ID] = [callRecord({ interview: "taken" })];
    render(<OnboardingPage />);
    const { socket, room } = await liveRoomCall();
    act(() => {
      room.ottoJoins();
    });

    endCall(socket, "finished");

    await waitFor(() => expect(room.isLeft).toBe(true));
  });

  it("loses the call when LiveKit cannot get its audio back, then reads what became of the words", async () => {
    callStates[CALL_ID] = [callRecord({ end_reason: "client_gone" })];
    render(<OnboardingPage />);
    const { socket, room } = await liveRoomCall();

    act(() => room.lose());

    expect(
      await screen.findByText("We lost the call. Saving what you said…"),
    ).toBeDefined();
    expect(socket.readyState).toBe(FakeWebSocket.CLOSED);
    expect(await screen.findByText("The connection dropped.")).toBeDefined();
    expect(statusReads).toContain(CALL_ID);
  });

  it("lets the call go on in the tab that took its room over", async () => {
    render(<OnboardingPage />);
    const { socket, room } = await liveRoomCall();

    act(() => room.lose(DISCONNECT_REASON.DUPLICATE_IDENTITY));

    expect(
      await screen.findByText(
        "Your call with Otto went on in another tab or on another device.",
      ),
    ).toBeDefined();
    expect(socket.readyState).toBe(FakeWebSocket.CLOSED);
    expect(statusReads).toEqual([]);
  });

  it.each([
    ["removed the browser", DISCONNECT_REASON.PARTICIPANT_REMOVED],
    ["deleted the room", DISCONNECT_REASON.ROOM_DELETED],
  ])(
    "shows the call's ending when the relay %s after the call ended",
    async (_case, reason) => {
      callStates[CALL_ID] = [callRecord({ interview: "taken" })];
      render(<OnboardingPage />);
      const { socket, room } = await liveRoomCall();
      act(() => {
        room.ottoJoins();
      });
      endCall(socket, "finished");

      act(() => room.lose(reason));

      expect(
        await screen.findByRole("button", { name: "Meet your team" }),
      ).toBeDefined();
      expect(room.isLeft).toBe(true);
      expect(
        screen.queryByText("We lost the call. Saving what you said…"),
      ).toBeNull();
    },
  );

  it.each([
    ["removes the browser", DISCONNECT_REASON.PARTICIPANT_REMOVED],
    ["deletes the room", DISCONNECT_REASON.ROOM_DELETED],
  ])(
    "waits for the socket's word when the relay %s before the call ends",
    async (_case, reason) => {
      callStates[CALL_ID] = [callRecord({ interview: "taken" })];
      render(<OnboardingPage />);
      const { socket, room } = await liveRoomCall();

      act(() => room.lose(reason));

      expect(room.isLeft).toBe(true);
      expect(screen.getByText("You're on the call with Otto")).toBeDefined();
      expect(socket.readyState).toBe(FakeWebSocket.OPEN);
      expect(screen.queryByRole("button", { name: "Rejoin" })).toBeNull();
      expect(
        screen.queryByText("We lost the call. Saving what you said…"),
      ).toBeNull();
      expect(statusReads).toEqual([]);

      endCall(socket, "finished");

      expect(
        await screen.findByRole("button", { name: "Meet your team" }),
      ).toBeDefined();
      expect(tracked("voice_interview_reconnected")).toEqual([]);
    },
  );

  it("keeps only its token for the call's room with the ticket, so a reload can join it again", async () => {
    render(<OnboardingPage />);
    await liveRoomCall();

    const kept = JSON.parse(sessionStorage.getItem(TICKET_KEY) ?? "{}");
    expect(kept.roomToken).toBe(ROOM_TOKEN);
    expect(kept).not.toHaveProperty("room");
    expect(JSON.stringify(kept)).not.toContain(ROOM_URL);
  });

  it("rejoins the call's room on the project the server names, with the token this tab kept, while the socket resumes", async () => {
    keepTicket(CALL_ID, ROOM_TOKEN);
    latestCall = liveRoomRecord();
    callStates[CALL_ID] = [latestCall];
    const letIn = fakes.holdRoomJoins();

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Rejoin" }));

    // The socket opens while the room's join is still under way.
    const socket = await relaySocketAfter();
    const room = await waitFor(() => {
      const made = fakes.room();
      if (!made?.token) throw new Error("No room joined yet");
      return made;
    });
    expect(room.published).toEqual([]);
    expect(room.url).toBe(ROOM_URL);
    expect(room.token).toBe(ROOM_TOKEN);
    act(() => socket.accept());
    expect(socket.messages()[0]).toEqual({ type: "resume", ticket: TICKET });
    act(() => letIn());
    await waitFor(() => expect(room.published).toHaveLength(1));
    act(() => socket.relay({ type: "resumed", gap_s: 4.1 }));
    expect(await screen.findByText("Reconnected")).toBeDefined();
    expect(mints).toBe(0);
  });

  it("joins no room address the tab's storage names on a rejoin", async () => {
    const elsewhere = "wss://elsewhere.example";
    keepTicket(CALL_ID, ROOM_TOKEN, {
      room: { url: elsewhere, token: ROOM_TOKEN },
      roomUrl: elsewhere,
      livekit_url: elsewhere,
    });
    latestCall = liveRoomRecord();
    callStates[CALL_ID] = [latestCall];

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Rejoin" }));

    await waitFor(() => expect(fakes.room()?.url).toBe(ROOM_URL));
    expect(fakes.rooms().map((room) => room.url)).toEqual([ROOM_URL]);
  });

  it("cannot rejoin a call when the server names no project for its room", async () => {
    keepTicket();
    latestCall = liveRoomRecord({ livekit_url: null });
    callStates[CALL_ID] = [latestCall];

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Rejoin" }));

    expect(
      await screen.findByText(
        "Your call with Otto is still going in another tab or on another device.",
      ),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Rejoin" })).toBeNull();
    expect(screen.getByRole("button", { name: "End that call" })).toBeDefined();
    expect(fakes.getUserMedia).not.toHaveBeenCalled();
    expect(fakes.sockets).toEqual([]);
    expect(fakes.rooms()).toEqual([]);
  });

  it("offers no Rejoin for a call this tab kept no room token for", async () => {
    keepTicket(CALL_ID, null);
    latestCall = liveRoomRecord();
    callStates[CALL_ID] = [latestCall];

    render(<OnboardingPage />);

    expect(
      await screen.findByRole("button", { name: "End that call" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Rejoin" })).toBeNull();
    expect(fakes.rooms()).toEqual([]);
  });

  it("offers Rejoin again when the call's room cannot be joined after a reload", async () => {
    keepTicket(CALL_ID, ROOM_TOKEN);
    latestCall = liveRoomRecord();
    callStates[CALL_ID] = [latestCall];
    fakes.refuseRoomJoins(new Error("the network is gone"));

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Rejoin" }));

    await waitFor(() => expect(fakes.rooms()).toHaveLength(1));
    expect(await screen.findByRole("button", { name: "Rejoin" })).toBeDefined();
    // The socket that opened alongside the join is let go with it.
    expect(fakes.sockets).toHaveLength(1);
    expect(fakes.sockets[0].readyState).toBe(FakeWebSocket.CLOSED);
    // A rejoin is no start: nothing was spent, and nothing ended.
    expect(tracked("voice_interview_start_failed")).toEqual([]);
    expect(ends).toEqual([]);
  });

  it("offers Rejoin again when the room fails after the socket resumed the call", async () => {
    keepTicket(CALL_ID, ROOM_TOKEN);
    latestCall = liveRoomRecord();
    callStates[CALL_ID] = [latestCall];
    const letIn = fakes.holdRoomJoins();

    render(<OnboardingPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Rejoin" }));
    const socket = await relaySocketAfter();
    act(() => socket.accept());
    act(() => socket.relay({ type: "resumed", gap_s: 2.5 }));
    expect(await screen.findByText("Reconnected")).toBeDefined();

    fakes.refuseRoomJoins(new Error("the network is gone"));
    act(() => letIn());

    expect(
      await screen.findByText("Your call with Otto is still going."),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Rejoin" })).toBeDefined();
    expect(socket.readyState).toBe(FakeWebSocket.CLOSED);
  });

  it("takes the microphone out of the room as soon as the person hangs up", async () => {
    render(<OnboardingPage />);
    const { socket, room } = await liveRoomCall();

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Hang up" }));

    expect(room.unpublished).toHaveLength(1);
    expect(socket.messages().filter((m) => m.type === "hangup")).toEqual([
      { type: "hangup" },
    ]);
  });

  it("leaves the room when the person types instead", async () => {
    render(<OnboardingPage />);
    const { room } = await liveRoomCall();

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Type instead" }));

    expect(await screen.findByText(GREETING)).toBeDefined();
    expect(room.isLeft).toBe(true);
  });

  it("leaves the room when the step goes away mid-call", async () => {
    const { unmount } = render(<OnboardingPage />);
    const { room } = await liveRoomCall();

    unmount();

    expect(room.isLeft).toBe(true);
  });
});
