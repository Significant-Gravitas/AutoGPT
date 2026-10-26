import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  createStreamHold,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import type { UIMessageChunk } from "ai";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CANCELLED_MARKER } from "../useCopilotStop";
import {
  clickStop,
  renderHost,
  resetChatRuntimes,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

// A message sent while the backend is still tearing down a stopped turn is
// queued as that turn's follow-up, and its row is written only later. The
// stream-end hydrate used to replace the chat with rows that did not have it
// yet, so the message vanished until a reload (SECRT-2843).

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

const FIRST = "Plan my week";
const AFTER_STOP = "Just Monday, please";
const LATER = "Thanks";
const LIVE_PARTIAL = "Looking at your calendar";
// Each stored row reads differently from what was streamed, so seeing it on
// screen means a hydrate has replaced the list. The stopped partial runs
// further because the backend kept generating until the cancel landed.
const STORED_PARTIAL = "Looking at your calendar, then your inbox";
const LATER_ANSWER = "You're welcome.";
const STORED_LATER_ANSWER = "You're welcome!";

const stored = { messages: [] as SessionDetailResponseMessagesItem[] };

beforeEach(() => {
  resetChatRuntimes();
  stored.messages = [];
});

afterEach(() => {
  resetChatRuntimes();
  server.resetHandlers();
});

describe("a message sent right after Stop", () => {
  it("stays in the chat through the hydrate, and appears once when its row lands", async () => {
    const posts: string[] = [];
    const { hold } = createStreamHold(4);
    server.use(
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
        async ({ request }) => {
          const { message = "" } = (await request.json()) as {
            message?: string;
          };
          posts.push(message);
          if (message === FIRST) {
            store("user", FIRST);
            return streamSseResponse(stoppableTurnChunks(), {
              hold,
              abortSignal: request.signal,
            });
          }
          if (message === AFTER_STOP) {
            // The stopped turn finishes tearing down while this send waits.
            store("assistant", STORED_PARTIAL);
            store("assistant", CANCELLED_MARKER);
            return streamSseResponse(followUpQueuedChunks(), {
              abortSignal: request.signal,
            });
          }
          store("user", message);
          store("assistant", STORED_LATER_ANSWER);
          return streamSseResponse(
            assistantTextChunks(LATER_ANSWER, {
              messageId: "later-answer",
              textPartId: "later-answer-text",
            }),
            { abortSignal: request.signal },
          );
        },
      ),
    );
    renderHost();
    // Registered after renderHost so the stored rows win over the empty session.
    server.use(storedSessionHandler());

    await typeAndSend(FIRST);
    await screen.findByText(LIVE_PARTIAL, undefined, { timeout: 5000 });
    await clickStop();
    await waitForTurnToSettle();

    await typeAndSend(AFTER_STOP);
    await screen.findByText(STORED_PARTIAL, undefined, { timeout: 5000 });
    await waitForTurnToSettle();
    expect(screen.getAllByText(AFTER_STOP)).toHaveLength(1);

    // The queued follow-up's row is written once the stopped turn is gone.
    store("user", AFTER_STOP);
    await typeAndSend(LATER);
    await screen.findByText(STORED_LATER_ANSWER, undefined, { timeout: 5000 });
    await waitFor(() => expect(posts).toEqual([FIRST, AFTER_STOP, LATER]));
    for (const text of [FIRST, AFTER_STOP, LATER]) {
      expect(screen.getAllByText(text)).toHaveLength(1);
    }
  }, 20_000);
});

function store(role: "user" | "assistant", content: string) {
  const sequence = stored.messages.length + 1;
  stored.messages = [
    ...stored.messages,
    {
      id: `db-${sequence}`,
      role,
      content,
      tool_call_id: null,
      tool_calls: null,
      sequence,
      duration_ms: null,
      created_at: `2026-05-13T00:${String(sequence).padStart(2, "0")}:00Z`,
      metadata: null,
    },
  ];
}

function storedSessionHandler() {
  return http.get(`*/api/chat/sessions/${TEST_SESSION_ID}`, () =>
    HttpResponse.json({
      id: TEST_SESSION_ID,
      created_at: "2026-05-13T00:00:00Z",
      updated_at: "2026-05-13T00:00:00Z",
      user_id: "test-user",
      chat_status: "idle",
      messages: stored.messages,
      has_more_messages: false,
      oldest_sequence: null,
      active_stream: null,
      metadata: { dry_run: false, builder_graph_id: null },
    }),
  );
}

// Held open after the first delta, so only the Stop click ends it.
function stoppableTurnChunks(): UIMessageChunk[] {
  return [
    { type: "start", messageId: "stopped-turn" },
    { type: "start-step" },
    { type: "text-start", id: "stopped-text" },
    // The trailing space lets the smoother release the last word.
    { type: "text-delta", id: "stopped-text", delta: `${LIVE_PARTIAL} ` },
    { type: "text-end", id: "stopped-text" },
    { type: "finish-step" },
    { type: "finish" },
  ];
}

// What `stream_chat_post` answers while a turn is still in flight: the send
// went to the pending buffer, and the stream carries nothing but that fact.
function followUpQueuedChunks(): UIMessageChunk[] {
  return [
    { type: "start", messageId: "queued-follow-up" },
    { type: "start-step" },
    { type: "data-follow-up-queued", data: {} },
    { type: "finish-step" },
    { type: "finish" },
  ];
}

async function waitForTurnToSettle() {
  await waitFor(
    () => expect(screen.queryByRole("button", { name: "Stop" })).toBeNull(),
    { timeout: 5000 },
  );
}
