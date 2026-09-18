import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { server } from "@/mocks/mock-server";
import { screen, waitFor } from "@testing-library/react";
import type { UIMessageChunk } from "ai";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import { useCopilotStreamStore } from "../copilotStreamStore";
import {
  renderHost,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
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

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

const EARLIER_PROMPT = "What is on my calendar";
const EARLIER_ANSWER = "You have two meetings today";
const RESUMED_PROMPT = "Book the afternoon slot";
// Persisted by the backend before the page reloaded, so it hydrates from the
// DB with an `<sessionId>-seq-N` id and every part already `state: "done"`.
const PERSISTED_HALF = "Checked the calendar for openings";
// Only ever produced by the GET-resume replay.
const REPLAYED_HALF = "Booked the three o clock slot";

const SSE_HEADERS = {
  "content-type": "text/event-stream",
  "cache-control": "no-cache",
  connection: "keep-alive",
  "x-vercel-ai-ui-message-stream": "v1",
  "x-accel-buffering": "no",
};

const parkedStreamReleases: Array<() => void> = [];

/**
 * SSE response that emits `chunks` and then parks instead of finishing — the
 * backend turn is still running, which is exactly the state a GET-resume
 * attaches to. Parking keeps `status === "streaming"` so
 * `useHydrateOnStreamEnd` never force-replaces the in-memory messages with
 * the DB snapshot, making the assertions below observe the resume path alone.
 */
function parkedSseResponse(chunks: UIMessageChunk[]) {
  const encoder = new TextEncoder();
  let index = 0;
  let released = false;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (released) {
        controller.close();
        return;
      }
      if (index < chunks.length) {
        const chunk = chunks[index++];
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`),
        );
        return;
      }
      return new Promise<void>((resolve) => {
        parkedStreamReleases.push(() => {
          released = true;
          try {
            controller.close();
          } catch {
            // already closed
          }
          resolve();
        });
      });
    },
  });
  return new HttpResponse(stream, { status: 200, headers: SSE_HEADERS });
}

function releaseParkedStreams() {
  while (parkedStreamReleases.length > 0) {
    parkedStreamReleases.pop()?.();
  }
}

function sessionMessage(
  sequence: number,
  role: "user" | "assistant",
  content: string,
): SessionDetailResponseMessagesItem {
  return {
    id: `db-${sequence}`,
    role,
    content,
    tool_call_id: null,
    tool_calls: null,
    sequence,
    duration_ms: null,
    created_at: `2026-05-13T00:0${sequence}:00Z`,
    metadata: null,
  };
}

/**
 * The turn the user reloaded into: one completed turn, then a user prompt
 * whose assistant answer is only half-persisted while the backend keeps
 * streaming it.
 */
const HYDRATED_SESSION_MESSAGES = [
  sessionMessage(1, "user", EARLIER_PROMPT),
  sessionMessage(2, "assistant", EARLIER_ANSWER),
  sessionMessage(3, "user", RESUMED_PROMPT),
  sessionMessage(4, "assistant", PERSISTED_HALF),
];

/**
 * The replay the backend sends on GET-resume: the whole active turn from its
 * start, so it re-emits the already-persisted half before the new tail. Two
 * separate text parts (not one concatenated blob) so each half renders as its
 * own element and can be counted exactly.
 */
const RESUME_REPLAY_CHUNKS: UIMessageChunk[] = [
  { type: "start", messageId: "resumed-turn-1" },
  { type: "start-step" },
  { type: "text-start", id: "replay-1" },
  { type: "text-delta", id: "replay-1", delta: PERSISTED_HALF },
  { type: "text-end", id: "replay-1" },
  { type: "text-start", id: "replay-2" },
  { type: "text-delta", id: "replay-2", delta: REPLAYED_HALF },
  { type: "text-end", id: "replay-2" },
];

/**
 * A turn the backend started on its own (the engine-switch continuation
 * dispatched with ``is_user_message=False``): the completed answer is
 * followed straight by the running turn's persisted half, with no user row
 * between them. Both rows are consecutive assistants, so hydration merges
 * them unless the active turn's ``started_at`` splits them apart.
 */
const BACKEND_STARTED_TURN_MESSAGES = [
  sessionMessage(1, "user", EARLIER_PROMPT),
  sessionMessage(2, "assistant", EARLIER_ANSWER),
  sessionMessage(3, "assistant", PERSISTED_HALF),
];

function renderResumedSession(
  messages: SessionDetailResponseMessagesItem[] = HYDRATED_SESSION_MESSAGES,
  activeStreamStartedAt = "2026-05-13T00:04:00Z",
) {
  let resumeRequests = 0;
  server.use(
    http.get(
      `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
      () => {
        resumeRequests += 1;
        return parkedSseResponse(RESUME_REPLAY_CHUNKS);
      },
    ),
  );
  renderHost({
    sessionOverride: {
      messages,
      active_stream: {
        turn_id: "turn-1",
        last_message_id: `db-${messages.length}`,
        started_at: activeStreamStartedAt,
      },
    },
  });
  return { getResumeRequests: () => resumeRequests };
}

beforeEach(() => {
  resetCopilotChatRegistry();
  // Message snapshots are keyed by session id in a module-level store, so
  // without this each case starts by rendering the previous case's
  // transcript and its assertions pass against stale DOM.
  useCopilotStreamStore.getState().resetAll();
});

afterEach(() => {
  releaseParkedStreams();
  resetCopilotChatRegistry();
});

describe("useCopilotStream — resume replays a db-hydrated turn", () => {
  it(
    "drops the hydrated partial so the replayed turn renders one bubble, not two",
    { timeout: 20000 },
    async () => {
      const { getResumeRequests } = renderResumedSession();

      // The hydrated half is on screen before the resume replaces it.
      expect(await screen.findByText(PERSISTED_HALF)).toBeDefined();

      await waitFor(() => expect(getResumeRequests()).toBe(1), {
        timeout: 10000,
      });
      // The replay's own tail — proves the GET-resume content actually landed.
      expect(
        await screen.findByText(REPLAYED_HALF, undefined, { timeout: 10000 }),
      ).toBeDefined();

      // The heart of the regression: the replay re-emits the persisted half,
      // so leaving the `-seq-N` hydrated message in place would show it twice
      // (two assistant bubbles, two tool chains) for one backend turn.
      expect(screen.getAllByText(PERSISTED_HALF)).toHaveLength(1);
    },
  );

  it(
    "trims only the resumed turn — the prompt and earlier turns survive",
    { timeout: 20000 },
    async () => {
      renderResumedSession();

      expect(
        await screen.findByText(REPLAYED_HALF, undefined, { timeout: 10000 }),
      ).toBeDefined();

      // The trim slices at the last user message, so that message and every
      // completed turn before it must stay — and stay exactly once.
      expect(screen.getAllByText(RESUMED_PROMPT)).toHaveLength(1);
      expect(screen.getAllByText(EARLIER_PROMPT)).toHaveLength(1);
      expect(screen.getAllByText(EARLIER_ANSWER)).toHaveLength(1);
    },
  );

  it(
    "keeps the completed answer when the running turn started without a user row",
    { timeout: 20000 },
    async () => {
      renderResumedSession(
        BACKEND_STARTED_TURN_MESSAGES,
        "2026-05-13T00:03:00Z",
      );

      expect(
        await screen.findByText(REPLAYED_HALF, undefined, { timeout: 10000 }),
      ).toBeDefined();

      // The answer the user already read belongs to the previous turn, which
      // the resume never replays — trimming back to the last user message
      // would erase it for the whole stream.
      expect(screen.getAllByText(EARLIER_ANSWER)).toHaveLength(1);
      expect(screen.getAllByText(EARLIER_PROMPT)).toHaveLength(1);
      // The running turn's persisted half is still trimmed, so the replay
      // does not double it.
      expect(screen.getAllByText(PERSISTED_HALF)).toHaveLength(1);
    },
  );
});
