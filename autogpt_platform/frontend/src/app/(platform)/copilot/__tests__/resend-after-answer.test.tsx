import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  createStreamHold,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import {
  clickStop,
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
  }, 15_000);

  it("forgets a send once it is stopped, so a reload cannot drop a resend", async () => {
    // The turn has opened its stream but not started answering when Stop lands.
    const { hold } = createStreamHold(1);
    server.use(
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
        ({ request }) =>
          streamSseResponse(assistantTextChunks(ANSWERS[0]), {
            hold,
            abortSignal: request.signal,
          }),
      ),
    );
    renderHost();

    await typeAndSend("yes");
    await waitFor(() => expect(lastSubmittedText()).toBe("yes"));
    await clickStop();
    await waitForTurnToSettle();

    // A turn cancelled before it streams leaves the stored chat ending on the
    // user's row, and this memory survives a reload in sessionStorage; kept,
    // it would mark the user's resend of "yes" as a double-click and drop it.
    expect(lastSubmittedText()).toBeNull();
  }, 15_000);

  it("forgets a send whose turn failed, so a reload cannot drop a resend", async () => {
    server.use(
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
        async () => {
          // Long enough to see the send remembered before the failure lands.
          await new Promise((resolve) => setTimeout(resolve, 300));
          return HttpResponse.json({ detail: "boom" }, { status: 500 });
        },
      ),
    );
    renderHost();

    await typeAndSend("yes");
    await waitFor(() => expect(lastSubmittedText()).toBe("yes"));

    await waitFor(() => expect(lastSubmittedText()).toBeNull(), {
      timeout: 5000,
    });
  }, 15_000);
});

function lastSubmittedText() {
  return useCopilotStreamStore.getState().getCoord(TEST_SESSION_ID)
    .lastSubmittedMessageText;
}
