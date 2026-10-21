import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  renderHost,
  resetChatRuntimes,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

// A user who answers "yes" to one question and then "yes" to the next. The
// duplicate guard used to swallow the second send for as long as the chat
// lived, with the composer already cleared (SECRT-2843).

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

const ANSWERS = ["Sure, go ahead.", "Done, both booked."];

// What the backend has persisted so far: the stream-end hydrate replaces the
// live list with these rows, exactly as it does in production.
const stored = { messages: [] as SessionDetailResponseMessagesItem[] };

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

function answeringStreamHandler(posts: string[]) {
  return http.post(
    `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
    async ({ request }) => {
      const body = (await request.json()) as { message?: string };
      posts.push(body.message ?? "");
      const turn = posts.length;
      const answer = ANSWERS[turn - 1] ?? ANSWERS[ANSWERS.length - 1];
      stored.messages = [
        ...stored.messages,
        sessionMessage(turn * 2 - 1, "user", body.message ?? ""),
        sessionMessage(turn * 2, "assistant", answer),
      ];
      return streamSseResponse(
        assistantTextChunks(answer, {
          messageId: `answer-${turn}`,
          textPartId: `answer-${turn}-text`,
        }),
        { abortSignal: request.signal },
      );
    },
  );
}

async function waitForTurnToSettle() {
  await waitFor(
    () => expect(screen.queryByRole("button", { name: "Stop" })).toBeNull(),
    { timeout: 5000 },
  );
}

beforeEach(() => {
  resetChatRuntimes();
  stored.messages = [];
});

afterEach(() => {
  resetChatRuntimes();
  server.resetHandlers();
});

describe("sending the same text twice in a row", () => {
  it('sends "yes" again once the first "yes" has been answered', async () => {
    const posts: string[] = [];
    server.use(answeringStreamHandler(posts));
    renderHost();
    // Registered after renderHost so the stored rows win over the empty session.
    server.use(storedSessionHandler());

    await typeAndSend("yes");
    await screen.findByText(ANSWERS[0], undefined, { timeout: 5000 });
    await waitForTurnToSettle();

    await typeAndSend("yes");

    await waitFor(() => expect(posts).toEqual(["yes", "yes"]), {
      timeout: 5000,
    });
    await screen.findByText(ANSWERS[1], undefined, { timeout: 5000 });
    expect(screen.getAllByText("yes")).toHaveLength(2);
  });
});
