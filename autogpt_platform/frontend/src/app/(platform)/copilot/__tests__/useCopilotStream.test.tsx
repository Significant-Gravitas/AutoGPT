import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getPostV2ProcessReviewActionMockHandler200 } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  copilotStreamHandler,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { folder } from "../components/ApprovalQueue/__tests__/fixtures";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import {
  renderHost,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

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

vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
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

const { toastMock } = vi.hoisted(() => ({ toastMock: vi.fn() }));

vi.mock("@/components/molecules/Toast/use-toast", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: toastMock };
});

// Records the `isFinishProbing` prop on every render so the producer-side
// lifecycle (set before the probe loop, reset in `finally`) is observable
// without reaching into `useCopilotStream` internals.
const { isFinishProbingHistory } = vi.hoisted(() => ({
  isFinishProbingHistory: [] as boolean[],
}));

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

function streamUrl() {
  return `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`;
}

function sessionJson(
  activeStream: { turn_id: string; last_message_id: string } | null,
): SessionDetailResponse {
  return {
    id: TEST_SESSION_ID,
    created_at: "2026-05-13T00:00:00Z",
    updated_at: "2026-05-13T00:00:00Z",
    user_id: "test-user",
    chat_status: "idle",
    messages: [],
    has_more_messages: false,
    oldest_sequence: null,
    active_stream: activeStream,
    metadata: { dry_run: false, builder_graph_id: null },
  };
}

beforeEach(() => {
  resetCopilotChatRegistry();
  isFinishProbingHistory.length = 0;
  toastMock.mockClear();
});

afterEach(() => {
  heldStreams.splice(0).forEach((release) => release());
  resetCopilotChatRegistry();
});

describe("useCopilotStream — isFinishProbing lifecycle", () => {
  it("sets the flag during the post-finish probe and resets it when no continuation is pending", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: TEST_BACKEND_BASE_URL,
        sessionId: TEST_SESSION_ID,
        chunks: assistantTextChunks("Hi."),
      }),
    );

    renderHost();
    await typeAndSend("hi");
    await screen.findByText("Hi.", undefined, { timeout: 5000 });

    // The probe loop runs for ~500ms after the SSE finish.
    await waitFor(() => expect(isFinishProbingHistory).toContain(true), {
      timeout: 5000,
    });
    // No active backend stream → the loop exits normally and the finally
    // block resets the flag.
    await waitFor(() => expect(isFinishProbingHistory.at(-1)).toBe(false), {
      timeout: 5000,
    });
  });

  it(
    "resets the flag via finally when the probe finds a continuation stream",
    { timeout: 15000 },
    async () => {
      // Flipped by the stream POST: the session GET only reports a live
      // continuation stream once the first turn has actually run.
      let continuationPending = false;
      let resumeRequested = false;

      renderHost();
      // Registered AFTER renderHost so these take precedence over the
      // default pinned handlers (MSW matches most-recently-added first).
      server.use(
        getGetV2GetSessionMockHandler200(() =>
          sessionJson(
            continuationPending
              ? { turn_id: "turn-2", last_message_id: "msg-2" }
              : null,
          ),
        ),
        http.post(streamUrl(), ({ request }) => {
          continuationPending = true;
          return streamSseResponse(assistantTextChunks("Hi."), {
            abortSignal: request.signal,
          });
        }),
        http.get(streamUrl(), ({ request }) => {
          resumeRequested = true;
          continuationPending = false;
          return streamSseResponse(
            assistantTextChunks(" continued", { messageId: "test-message-2" }),
            { abortSignal: request.signal },
          );
        }),
      );

      await typeAndSend("hi");
      await screen.findByText("Hi.", undefined, { timeout: 5000 });

      // Probe window: the flag is up while the active-stream probe runs.
      await waitFor(() => expect(isFinishProbingHistory).toContain(true), {
        timeout: 5000,
      });
      // The probe sees the continuation stream → early-return through the
      // reconnect path → the finally block must still reset the flag.
      await waitFor(() => expect(isFinishProbingHistory.at(-1)).toBe(false), {
        timeout: 5000,
      });
      // ...and the reconnect actually picked the continuation turn up.
      await waitFor(() => expect(resumeRequested).toBe(true), {
        timeout: 8000,
      });
    },
  );
});

describe("useCopilotStream — a turn the server chains onto the one that ended", () => {
  it(
    "attaches to the chained turn once, without a connection-lost toast",
    { timeout: 15000 },
    async () => {
      let activeTurn: string | null = "turn-1";
      const resumedTurns: (string | null)[] = [];
      server.use(
        http.get(streamUrl(), () => {
          const turn = activeTurn;
          resumedTurns.push(turn);
          if (turn === "turn-1") {
            return turnResponse("First.", "m-1", () => {
              activeTurn = "turn-2";
            });
          }
          return turnResponse("Second.", "m-2", () => {
            activeTurn = null;
          });
        }),
      );
      renderHost({ sessionResponse: liveSession(() => activeTurn) });

      // A reconnect would toast before its resume, so the turn's text
      // appearing means the decision has been made.
      await screen.findByText("Second.", undefined, { timeout: 5000 });

      expect(resumedTurns).toEqual(["turn-1", "turn-2"]);
      expect(connectionLostToasts()).toBe(0);
    },
  );

  it(
    "still reconnects with a toast when a stream ends while its own turn runs on",
    { timeout: 15000 },
    async () => {
      let activeTurn: string | null = "turn-1";
      const resumedTurns: (string | null)[] = [];
      server.use(
        http.get(streamUrl(), () => {
          resumedTurns.push(activeTurn);
          if (resumedTurns.length === 1) {
            // Cut off before its finish: the backend turn is still running.
            return streamSseResponse(
              assistantTextChunks("Partial", { messageId: "m-1" }).slice(0, 5),
            );
          }
          return turnResponse("Partial, then done.", "m-1", () => {
            activeTurn = null;
          });
        }),
      );
      renderHost({ sessionResponse: liveSession(() => activeTurn) });

      await screen.findByText("Partial, then done.", undefined, {
        timeout: 8000,
      });

      expect(resumedTurns).toEqual(["turn-1", "turn-1"]);
      expect(connectionLostToasts()).toBe(1);
    },
  );

  it.each(["resumed", "sent"] as const)(
    "an approval answered mid-stream and the turn it chains on share one resume (%s turn)",
    async (start) => {
      let activeTurn: string | null = start === "resumed" ? "turn-1" : null;
      let approved = false;
      const resumedTurns: (string | null)[] = [];
      const turn1 = hold();
      function firstTurn() {
        // The trailing space lets the POST path's smoother release the word.
        return turnResponse(
          "Working on it ",
          "m-1",
          () => {
            activeTurn = "turn-2";
          },
          turn1.released,
        );
      }
      server.use(
        http.get("*/api/review/session/:sessionId", () =>
          HttpResponse.json(approved ? [] : [folder("a", "Q3 reports")]),
        ),
        getPostV2ProcessReviewActionMockHandler200(() => {
          approved = true;
          return { approved_count: 1, rejected_count: 0, failed_count: 0 };
        }),
        http.post(streamUrl(), () => {
          activeTurn = "turn-1";
          return firstTurn();
        }),
        http.get(streamUrl(), () => {
          resumedTurns.push(activeTurn);
          if (activeTurn === "turn-1") return firstTurn();
          // Held open, so both triggers land while it is still streaming.
          return turnResponse(
            "Approved and done.",
            "m-2",
            () => {
              activeTurn = null;
            },
            hold().released,
          );
        }),
      );
      renderHost({ sessionResponse: liveSession(() => activeTurn) });
      if (start === "sent") await typeAndSend("Make the folder");

      await screen.findByText("Working on it", undefined, { timeout: 5000 });
      await userEvent
        .setup()
        .click(await screen.findByRole("button", { name: "Approve" }));
      await waitFor(() => expect(approved).toBe(true));
      turn1.release();

      await screen.findByText("Approved and done.", undefined, {
        timeout: 5000,
      });
      await waitForProbeToSettle();
      // Past the 1 s reconnect delay, which is where the old second resume came from.
      await new Promise((resolve) => setTimeout(resolve, 1500));

      expect(resumedTurns.filter((turn) => turn === "turn-2")).toHaveLength(1);
    },
    15000,
  );

  it(
    "a stalled restore is still replaced, and the stream it replaced ending late starts nothing",
    { timeout: 20000 },
    async () => {
      let resumes = 0;
      const stalled = hold();
      server.use(
        http.get(streamUrl(), () => {
          resumes += 1;
          if (resumes === 1) return stalledResponse(stalled.released);
          return turnResponse("Recovered.", "m-1", () => {}, hold().released);
        }),
      );
      renderHost({ sessionResponse: liveSession(() => "turn-1") });

      // The 6 s restore watchdog reconnects past the resume that never streamed.
      await screen.findByText("Recovered.", undefined, { timeout: 12000 });
      stalled.release();
      await waitForProbeToSettle();
      await new Promise((resolve) => setTimeout(resolve, 2500));

      expect(resumes).toBe(2);
    },
  );
});

const heldStreams: (() => void)[] = [];

function hold() {
  let release = () => {};
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  heldStreams.push(release);
  return { released, release };
}

const SSE_HEADERS = {
  "content-type": "text/event-stream",
  "x-vercel-ai-ui-message-stream": "v1",
};

function liveSession(activeTurn: () => string | null) {
  return getGetV2GetSessionMockHandler200(() => {
    const turn = activeTurn();
    return sessionJson(turn ? { turn_id: turn, last_message_id: "0-0" } : null);
  });
}

/** A turn's replay whose finish waits for `released`; `onEnd` runs just before
 *  it, where the backend wakes the next turn. */
function turnResponse(
  text: string,
  messageId: string,
  onEnd: () => void,
  released: Promise<void> = Promise.resolve(),
) {
  const chunks = assistantTextChunks(text, { messageId });
  async function* frames() {
    yield* chunks.slice(0, 4);
    await released;
    onEnd();
    yield* chunks.slice(4);
  }
  const iterator = frames();
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      const next = await iterator.next();
      if (next.done) {
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
        return;
      }
      controller.enqueue(
        encoder.encode(`data: ${JSON.stringify(next.value)}\n\n`),
      );
    },
  });
  return new HttpResponse(stream, { status: 200, headers: SSE_HEADERS });
}

function stalledResponse(released: Promise<void>) {
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      await released;
      controller.close();
    },
  });
  return new HttpResponse(stream, { status: 200, headers: SSE_HEADERS });
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

function connectionLostToasts() {
  return toastMock.mock.calls.filter(
    ([options]) => options?.title === "Connection lost",
  ).length;
}
