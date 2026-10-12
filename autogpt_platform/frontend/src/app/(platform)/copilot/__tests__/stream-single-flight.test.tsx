import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { server } from "@/mocks/mock-server";
import { assistantTextChunks } from "@/tests/integrations/copilot-sse";
import { getPostV2ProcessReviewActionMockHandler200 } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { heldRead } from "../components/ApprovalQueue/__tests__/fixtures";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import {
  renderHost,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
  waitForInputReady,
} from "./sse-helpers";

// One live stream per chat. A second resume started beside a live request
// made the AI SDK's request cleanup read a response the other one had already
// cleared (Sentry BUILDER-777) and drop that turn's end-of-turn handling.

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

const { authGate } = vi.hoisted(() => ({
  authGate: { held: null as Promise<void> | null },
}));

vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => {
      const held = authGate.held;
      authGate.held = null;
      if (held) await held;
      return { "x-test-auth": "yes" };
    },
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isUserLoading: false, isLoggedIn: true }),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT",
  },
  useGetFlag: () => false,
}));

const { toastMock, isFinishProbingHistory } = vi.hoisted(() => ({
  toastMock: vi.fn(),
  isFinishProbingHistory: [] as boolean[],
}));

vi.mock("@/components/molecules/Toast/use-toast", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: toastMock };
});

vi.mock("../useHydrateOnStreamEnd", async (importActual) => {
  const actual =
    await importActual<typeof import("../useHydrateOnStreamEnd")>();
  return {
    ...actual,
    useHydrateOnStreamEnd: (
      args: Parameters<typeof actual.useHydrateOnStreamEnd>[0],
    ) => {
      isFinishProbingHistory.push(args.isFinishProbing);
      return actual.useHydrateOnStreamEnd(args);
    },
  };
});

const STREAM_URL = `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`;

let openStreams = 0;
let maxOpenStreams = 0;
let consoleError: ReturnType<typeof vi.spyOn>;
const releases: (() => void)[] = [];

beforeEach(() => {
  resetCopilotChatRegistry();
  openStreams = 0;
  maxOpenStreams = 0;
  isFinishProbingHistory.length = 0;
  toastMock.mockClear();
  consoleError = vi.spyOn(console, "error");
});

afterEach(() => {
  authGate.held = null;
  releases.splice(0).forEach((release) => release());
  consoleError.mockRestore();
  vi.useRealTimers();
  resetCopilotChatRegistry();
});

describe("copilot stream — one live stream per chat", () => {
  it(
    "a tab wake while the turn is streaming keeps the live stream",
    { timeout: 15000 },
    async () => {
      let activeTurn: string | null = null;
      let resumes = 0;
      const turn = hold();
      server.use(
        http.post(STREAM_URL, ({ request }) => {
          activeTurn = "turn-1";
          return liveTurn({
            text: "Counted to three.",
            signal: request.signal,
            released: turn.released,
            onEnd: () => {
              activeTurn = null;
            },
          });
        }),
        http.get(STREAM_URL, ({ request }) => {
          resumes += 1;
          return liveTurn({
            text: "Counted to three.",
            signal: request.signal,
            onEnd: () => {
              activeTurn = null;
            },
          });
        }),
      );
      renderHost({ sessionResponse: liveSession(() => activeTurn) });

      await typeAndSend("count");
      await screen.findByText(/Working on it/, undefined, { timeout: 5000 });
      await wakeAfter(31_000);
      await settle();

      turn.release();
      await screen.findByText(/Counted to three\./, undefined, {
        timeout: 5000,
      });
      await waitForProbeToSettle();

      expect(resumes).toBe(0);
      expect(maxOpenStreams).toBe(1);
      expect(sdkCleanupCrashes()).toEqual([]);
      expect(screen.getAllByText(/Counted to three\./)).toHaveLength(1);
    },
  );

  it(
    "a tab wake after the stream went quiet replaces it rather than running beside it",
    { timeout: 15000 },
    async () => {
      let activeTurn: string | null = null;
      let resumes = 0;
      server.use(
        http.post(STREAM_URL, ({ request }) => {
          activeTurn = "turn-1";
          // Never finishes on its own: the connection died while asleep.
          return liveTurn({
            text: "never",
            signal: request.signal,
            released: hold().released,
          });
        }),
        http.get(STREAM_URL, ({ request }) => {
          resumes += 1;
          return liveTurn({
            text: "Back after the wake.",
            signal: request.signal,
            onEnd: () => {
              activeTurn = null;
            },
          });
        }),
      );
      renderHost({ sessionResponse: liveSession(() => activeTurn) });

      await typeAndSend("count");
      await screen.findByText(/Working on it/, undefined, { timeout: 5000 });
      await wakeAfter(90_000);

      await screen.findByText(/Back after the wake\./, undefined, {
        timeout: 5000,
      });
      await waitForProbeToSettle();

      expect(resumes).toBe(1);
      expect(maxOpenStreams).toBe(1);
      expect(sdkCleanupCrashes()).toEqual([]);
      expect(finishProbes()).toBe(1);
    },
  );

  it(
    "a read released to Otto mid-turn attaches to the turn it chains on once",
    { timeout: 15000 },
    async () => {
      // The nightly canary's BUILDER-777 on dev-builder: the held read was
      // released while the turn streamed; when that turn ended, the follow
      // path and the end-of-turn probe each resumed the chained turn, and the
      // two replays crashed the SDK's cleanup when they finished together.
      let activeTurn: string | null = null;
      let released = false;
      const resumedTurns: (string | null)[] = [];
      const turn1 = hold();
      server.use(
        http.get("*/api/review/session/:sessionId", () =>
          HttpResponse.json(released ? [] : [heldRead("r", "a.example")]),
        ),
        getPostV2ProcessReviewActionMockHandler200(() => {
          released = true;
          return { approved_count: 1, rejected_count: 0, failed_count: 0 };
        }),
        http.post(STREAM_URL, ({ request }) => {
          activeTurn = "turn-1";
          return liveTurn({
            text: "Read it.",
            signal: request.signal,
            released: turn1.released,
            onEnd: () => {
              activeTurn = "turn-2";
            },
          });
        }),
        http.get(STREAM_URL, ({ request }) => {
          resumedTurns.push(activeTurn);
          return liveTurn({
            text: "Summarised the page.",
            signal: request.signal,
            released: hold().released,
            onEnd: () => {
              activeTurn = null;
            },
          });
        }),
      );
      renderHost({ sessionResponse: liveSession(() => activeTurn) });

      await typeAndSend("read a.example");
      await screen.findByText(/Working on it/, undefined, { timeout: 5000 });
      await userEvent
        .setup()
        .click(await screen.findByRole("button", { name: "Release to Otto" }));
      await waitFor(() => expect(released).toBe(true));
      turn1.release();

      await waitFor(() => expect(resumedTurns).toEqual(["turn-2"]), {
        timeout: 5000,
      });
      // Past the post-finish probe and the 1 s reconnect delay.
      await waitForProbeToSettle();
      await act(() => new Promise((resolve) => setTimeout(resolve, 1500)));
      releases.splice(0).forEach((release) => release());

      await screen.findByText(/Summarised the page\./, undefined, {
        timeout: 5000,
      });
      expect(resumedTurns).toEqual(["turn-2"]);
      expect(maxOpenStreams).toBe(1);
      expect(sdkCleanupCrashes()).toEqual([]);
    },
  );

  it(
    "a resume replaced while it is still preparing never reaches the network",
    { timeout: 30000 },
    async () => {
      // The restore's resume waits on its auth headers; the restore watchdog
      // replaces it meanwhile. Its abort must land before it ever fetches.
      const auth = hold();
      authGate.held = auth.released;
      let resumes = 0;
      server.use(
        http.get(STREAM_URL, ({ request }) => {
          resumes += 1;
          return liveTurn({
            text: "Restored once.",
            signal: request.signal,
            released: hold().released,
          });
        }),
      );
      renderHost({ sessionResponse: liveSession(() => "turn-1") });

      // Restore watchdog (6 s) and reconnect delay (1 s); the replaced
      // request ends at once, though its headers never came.
      await screen.findByText(/Working on it/, undefined, { timeout: 12000 });
      auth.release();
      await act(() => new Promise((resolve) => setTimeout(resolve, 1000)));

      expect(resumes).toBe(1);
      expect(maxOpenStreams).toBe(1);
      expect(sdkCleanupCrashes()).toEqual([]);
    },
  );

  it(
    "a tab wake while a send is still preparing leaves the send alone",
    { timeout: 15000 },
    async () => {
      // The send waits on its auth headers, so its fetch has not run and no
      // byte has arrived; the wake must still see a live request, not a
      // quiet one to replace.
      let activeTurn: string | null = null;
      let resumes = 0;
      server.use(
        http.post(STREAM_URL, ({ request }) =>
          liveTurn({
            text: "Sent once.",
            signal: request.signal,
            onEnd: () => {
              activeTurn = null;
            },
          }),
        ),
        http.get(STREAM_URL, ({ request }) => {
          resumes += 1;
          return liveTurn({ text: "replayed", signal: request.signal });
        }),
      );
      renderHost({ sessionResponse: liveSession(() => activeTurn) });
      await waitForInputReady();
      const auth = hold();
      authGate.held = auth.released;

      await typeAndSend("count");
      activeTurn = "turn-1";
      await wakeAfter(31_000);
      auth.release();

      await screen.findByText(/Sent once\./, undefined, { timeout: 5000 });
      await waitForProbeToSettle();

      expect(resumes).toBe(0);
      expect(maxOpenStreams).toBe(1);
      expect(sdkCleanupCrashes()).toEqual([]);
    },
  );

  it(
    "a send stuck in its auth lookup past the watchdog is still sent, once",
    { timeout: 20000 },
    async () => {
      // The watchdog's reconnect replaces the send while it waits on its
      // headers; nothing was posted, so its GET finds no turn (204). The send
      // must not vanish with it.
      vi.useFakeTimers({
        toFake: [
          "setTimeout",
          "clearTimeout",
          "setInterval",
          "clearInterval",
          "Date",
        ],
        shouldAdvanceTime: true,
      });
      let posts = 0;
      let resumes = 0;
      server.use(
        http.post(STREAM_URL, ({ request }) => {
          posts += 1;
          return liveTurn({ text: "Sent after all.", signal: request.signal });
        }),
        http.get(STREAM_URL, () => {
          resumes += 1;
          return new HttpResponse(null, { status: 204 });
        }),
      );
      renderHost({ sessionResponse: liveSession(() => null) });
      await waitForInputReady();
      const auth = hold();
      authGate.held = auth.released;

      await typeAndSend("count");
      await act(() => vi.advanceTimersByTimeAsync(61_000));
      await act(() => vi.advanceTimersByTimeAsync(1_500));

      await screen.findByText(/Sent after all\./, undefined, {
        timeout: 5000,
      });
      auth.release();
      await act(() => vi.advanceTimersByTimeAsync(2_000));

      expect(resumes).toBe(1);
      expect(posts).toBe(1);
      expect(screen.getAllByText("count")).toHaveLength(1);
      expect(sdkCleanupCrashes()).toEqual([]);
    },
  );

  it(
    "a retried send the server refuses with 402 gives its text back once",
    { timeout: 20000 },
    async () => {
      // The lost-send retry and the paywall refusal both give text back. The
      // retry reached the network, so only the refusal may; the composer
      // must hold the text once and the user is not told it went unsent.
      vi.useFakeTimers({
        toFake: [
          "setTimeout",
          "clearTimeout",
          "setInterval",
          "clearInterval",
          "Date",
        ],
        shouldAdvanceTime: true,
      });
      let posts = 0;
      let resumes = 0;
      server.use(
        http.post(STREAM_URL, () => {
          posts += 1;
          return HttpResponse.json(
            { detail: "A plan is required to send" },
            { status: 402 },
          );
        }),
        http.get(STREAM_URL, () => {
          resumes += 1;
          return new HttpResponse(null, { status: 204 });
        }),
      );
      renderHost({ sessionResponse: liveSession(() => null) });
      await waitForInputReady();
      const auth = hold();
      authGate.held = auth.released;

      await typeAndSend("count to three");
      await act(() => vi.advanceTimersByTimeAsync(61_000));
      await act(() => vi.advanceTimersByTimeAsync(1_500));

      const input = screen.getByLabelText(
        /chat message input/i,
      ) as HTMLTextAreaElement;
      await waitFor(() => expect(input.value).toBe("count to three"));
      auth.release();
      await act(() => vi.advanceTimersByTimeAsync(2_000));

      expect(posts).toBe(1);
      expect(resumes).toBe(1);
      expect(input.value).toBe("count to three");
      // The refused bubble is gone; only the composer holds the text.
      expect(
        screen.queryAllByText("count to three", { ignore: "textarea" }),
      ).toHaveLength(0);
      expect(
        toastMock.mock.calls.filter(
          ([options]) => options?.title === "Message not sent",
        ),
      ).toHaveLength(0);
      expect(sdkCleanupCrashes()).toEqual([]);
    },
  );

  it(
    "heartbeats keep a quiet stream alive past the stall watchdog",
    { timeout: 20000 },
    async () => {
      vi.useFakeTimers({
        toFake: [
          "setTimeout",
          "clearTimeout",
          "setInterval",
          "clearInterval",
          "Date",
        ],
        shouldAdvanceTime: true,
      });
      let activeTurn: string | null = null;
      let resumes = 0;
      const quiet = hold();
      server.use(
        http.post(STREAM_URL, ({ request }) => {
          activeTurn = "turn-1";
          return liveTurn({
            text: "Done after a long tool call.",
            signal: request.signal,
            released: quiet.released,
            heartbeats: true,
            onEnd: () => {
              activeTurn = null;
            },
          });
        }),
        http.get(STREAM_URL, ({ request }) => {
          resumes += 1;
          return liveTurn({ text: "replayed", signal: request.signal });
        }),
      );
      renderHost({ sessionResponse: liveSession(() => activeTurn) });

      await typeAndSend("run the long tool");
      await screen.findByText(/Working on it/, undefined, { timeout: 5000 });
      // 90 s with nothing new on screen, a heartbeat every 15 s.
      for (let elapsed = 0; elapsed < 90_000; elapsed += 5_000) {
        await act(() => vi.advanceTimersByTimeAsync(5_000));
      }
      quiet.release();

      await screen.findByText(/Done after a long tool call\./, undefined, {
        timeout: 5000,
      });
      expect(connectionLostToasts()).toBe(0);
      expect(resumes).toBe(0);
      expect(maxOpenStreams).toBe(1);
    },
  );

  it(
    "a stalled live stream is replaced once, and its turn ends once",
    { timeout: 20000 },
    async () => {
      vi.useFakeTimers({
        toFake: [
          "setTimeout",
          "clearTimeout",
          "setInterval",
          "clearInterval",
          "Date",
        ],
        shouldAdvanceTime: true,
      });
      let activeTurn: string | null = null;
      let resumes = 0;
      server.use(
        http.post(STREAM_URL, ({ request }) => {
          activeTurn = "turn-1";
          // Open, but nothing arrives: no content and no heartbeats.
          return liveTurn({
            text: "never",
            signal: request.signal,
            released: hold().released,
          });
        }),
        http.get(STREAM_URL, ({ request }) => {
          resumes += 1;
          return liveTurn({
            text: "Recovered the turn.",
            signal: request.signal,
            onEnd: () => {
              activeTurn = null;
            },
          });
        }),
      );
      renderHost({ sessionResponse: liveSession(() => activeTurn) });

      await typeAndSend("count");
      await screen.findByText(/Working on it/, undefined, { timeout: 5000 });
      await act(() => vi.advanceTimersByTimeAsync(61_000));
      await act(() => vi.advanceTimersByTimeAsync(1_500));

      await screen.findByText(/Recovered the turn\./, undefined, {
        timeout: 5000,
      });
      await waitForProbeToSettle();

      expect(resumes).toBe(1);
      expect(maxOpenStreams).toBe(1);
      expect(sdkCleanupCrashes()).toEqual([]);
      expect(finishProbes()).toBe(1);
      expect(screen.getAllByText(/Recovered the turn\./)).toHaveLength(1);
    },
  );
});

function hold() {
  let release = () => {};
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  releases.push(release);
  return { released, release };
}

const SSE_HEADERS = {
  "content-type": "text/event-stream",
  "x-vercel-ai-ui-message-stream": "v1",
};

const HEARTBEAT_EVERY_MS = 15_000;

/**
 * A turn as the backend streams it: its opening text right away, then the
 * rest once `released` resolves. With `heartbeats`, the quiet middle carries
 * `: heartbeat` comments, as the stream registry sends while a turn is silent.
 * Counted as open until it ends or the client aborts it.
 */
function liveTurn({
  text,
  signal,
  released = Promise.resolve(),
  heartbeats = false,
  onEnd = () => {},
}: {
  text: string;
  signal: AbortSignal;
  released?: Promise<void>;
  heartbeats?: boolean;
  onEnd?: () => void;
}) {
  const opening = [
    ...assistantTextChunks("Working on it. ").slice(0, 4),
    { type: "text-end", id: "test-text-1" },
  ];
  const rest = assistantTextChunks(text, { textPartId: "test-text-2" }).slice(
    2,
  );
  const encoder = new TextEncoder();
  let closed = false;
  openStreams += 1;
  maxOpenStreams = Math.max(maxOpenStreams, openStreams);
  function close() {
    if (closed) return;
    closed = true;
    openStreams -= 1;
  }
  const aborted = new Promise<void>((resolve) => {
    if (signal.aborted) resolve();
    signal.addEventListener("abort", () => resolve(), { once: true });
  });
  let heartbeatTimer: ReturnType<typeof setInterval> | undefined;
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      function send(frame: string) {
        if (!closed) controller.enqueue(encoder.encode(frame));
      }
      opening.forEach((chunk) => send(`data: ${JSON.stringify(chunk)}\n\n`));
      if (heartbeats) {
        heartbeatTimer = setInterval(
          () => send(": heartbeat\n\n"),
          HEARTBEAT_EVERY_MS,
        );
      }
      await Promise.race([released, aborted]);
      clearInterval(heartbeatTimer);
      if (signal.aborted) {
        close();
        return;
      }
      onEnd();
      rest.forEach((chunk) => send(`data: ${JSON.stringify(chunk)}\n\n`));
      send("data: [DONE]\n\n");
      close();
      controller.close();
    },
    cancel() {
      clearInterval(heartbeatTimer);
      close();
    },
  });
  return new HttpResponse(stream, { status: 200, headers: SSE_HEADERS });
}

function liveSession(activeTurn: () => string | null) {
  return getGetV2GetSessionMockHandler200(() => {
    const turn = activeTurn();
    return {
      id: TEST_SESSION_ID,
      created_at: "2026-05-13T00:00:00Z",
      updated_at: "2026-05-13T00:00:00Z",
      user_id: "test-user",
      chat_status: "idle",
      messages: [],
      has_more_messages: false,
      oldest_sequence: null,
      active_stream: turn ? { turn_id: turn, last_message_id: "0-0" } : null,
      metadata: { dry_run: false, builder_graph_id: null },
    } satisfies SessionDetailResponse;
  });
}

/** Hide the tab, let `hiddenForMs` pass, and show it again. */
async function wakeAfter(hiddenForMs: number) {
  const realNow = Date.now.bind(Date);
  let visibility: DocumentVisibilityState = "hidden";
  const visibilitySpy = vi
    .spyOn(document, "visibilityState", "get")
    .mockImplementation(() => visibility);
  await act(async () => {
    document.dispatchEvent(new Event("visibilitychange"));
  });
  const nowSpy = vi
    .spyOn(Date, "now")
    .mockImplementation(() => realNow() + hiddenForMs);
  visibility = "visible";
  await act(async () => {
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await settle();
  visibilitySpy.mockRestore();
  nowSpy.mockRestore();
}

async function settle() {
  await act(() => new Promise((resolve) => setTimeout(resolve, 300)));
}

async function waitForProbeToSettle() {
  await waitFor(
    () => {
      expect(isFinishProbingHistory).toContain(true);
      expect(isFinishProbingHistory.at(-1)).toBe(false);
    },
    { timeout: 5000 },
  );
}

/** How many times the end-of-turn probe started. */
function finishProbes() {
  return isFinishProbingHistory.filter(
    (value, i) => value && !isFinishProbingHistory[i - 1],
  ).length;
}

/** The AI SDK logs BUILDER-777 from its request cleanup with console.error. */
function sdkCleanupCrashes() {
  return consoleError.mock.calls.filter(
    ([error]: unknown[]) =>
      error instanceof TypeError && error.message.includes("reading 'state'"),
  );
}

function connectionLostToasts() {
  return toastMock.mock.calls.filter(
    ([options]) => options?.title === "Connection lost",
  ).length;
}
