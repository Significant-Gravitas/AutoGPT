import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  renderHost,
  resetChatRuntimes,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
} from "./sse-helpers";

// A follow-up typed while Otto was busy waits in the pending queue for the
// running turn to read it. When that turn ends first, nothing will answer it:
// the chat must say so and offer to send it again, after a reload too, rather
// than show it "Queued" for ever (SECRT-3156).

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

const FOLLOW_UP = "What is my team working on right now?";
const ANSWER = "Maya is drafting the launch plan.";

const PENDING_URL = `*/api/chat/sessions/${TEST_SESSION_ID}/messages/pending`;
const STREAM_URL = `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`;

const stored = { messages: [] as SessionDetailResponseMessagesItem[] };
const buffer = { messages: [] as string[] };

beforeEach(() => {
  resetChatRuntimes();
  stored.messages = [];
  store("user", "Plan my week");
  store("assistant", "Here is your week.");
  // The turn ended on an error with the follow-up still queued.
  buffer.messages = [FOLLOW_UP];
});

afterEach(() => {
  resetChatRuntimes();
  server.resetHandlers();
});

describe("a follow-up the last turn ended without reading", () => {
  it("shows as not sent, with why and a Retry, instead of Queued", async () => {
    renderChat();

    expect(
      await screen.findByText("Didn't go through", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    expect(
      screen.getByText("Otto hit a problem before reading it."),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: /retry/i })).toBeDefined();
    expect(screen.queryByText("Queued")).toBeNull();
    // Once, and not as a sent bubble: the server has no row for it.
    expect(screen.getAllByText(FOLLOW_UP)).toHaveLength(1);
  });

  it("goes out once on Retry and gets its answer", async () => {
    const posts: string[] = [];
    server.use(
      http.post(STREAM_URL, async ({ request }) => {
        const { message = "" } = (await request.json()) as {
          message?: string;
        };
        posts.push(message);
        // The server takes the held copy out of the queue and stores the row.
        buffer.messages = [];
        store("user", message);
        store("assistant", ANSWER);
        return streamSseResponse(
          assistantTextChunks(ANSWER, {
            messageId: "retry-answer",
            textPartId: "retry-answer-text",
          }),
          { abortSignal: request.signal },
        );
      }),
    );
    renderChat();
    const retry = await screen.findByRole(
      "button",
      { name: /retry/i },
      { timeout: 5000 },
    );

    await userEvent.setup().click(retry);

    await screen.findByText(ANSWER, undefined, { timeout: 5000 });
    expect(posts).toEqual([FOLLOW_UP]);
    await waitFor(() =>
      expect(screen.queryByText("Didn't go through")).toBeNull(),
    );
    expect(screen.getAllByText(FOLLOW_UP)).toHaveLength(1);
  }, 15_000);
});

function renderChat() {
  renderHost({
    sessionOverride: { messages: stored.messages },
    pendingMessages: buffer.messages,
    pendingHeldReason: "error",
  });
  // Registered after renderHost, for the reads after the first.
  server.use(
    http.get(`*/api/chat/sessions/${TEST_SESSION_ID}`, () =>
      HttpResponse.json({
        id: TEST_SESSION_ID,
        created_at: "2026-10-23T00:00:00Z",
        updated_at: "2026-10-23T00:00:00Z",
        user_id: "test-user",
        chat_status: "idle",
        messages: stored.messages,
        has_more_messages: false,
        oldest_sequence: null,
        active_stream: null,
        metadata: { dry_run: false, builder_graph_id: null },
      }),
    ),
    http.get(PENDING_URL, () =>
      HttpResponse.json({
        messages: buffer.messages,
        count: buffer.messages.length,
        held_reason: buffer.messages.length > 0 ? "error" : null,
      }),
    ),
  );
}

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
      created_at: `2026-10-23T00:${String(sequence).padStart(2, "0")}:00Z`,
      metadata: null,
    },
  ];
}
