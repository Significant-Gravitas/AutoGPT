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
  typeAndSend,
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

const streamPath = vi.hoisted(() => ({ runtime: false }));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === "copilot-stream-runtime" ? streamPath.runtime : false,
  };
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
  streamPath.runtime = false;
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

  it("sends one message when two held follow-ups are retried at once", async () => {
    const SECOND = "And Monday?";
    buffer.messages = [FOLLOW_UP, SECOND];
    const posts: string[] = [];
    const queued: string[] = [];
    server.use(
      http.post(STREAM_URL, async ({ request }) => {
        const { message = "" } = (await request.json()) as {
          message?: string;
        };
        posts.push(message);
        // The new turn reads every held follow-up at its start.
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
      http.post(PENDING_URL, async ({ request }) => {
        queued.push(((await request.json()) as { message: string }).message);
        return HttpResponse.json({
          buffer_length: 1,
          max_buffer_length: 10,
          turn_in_flight: true,
        });
      }),
    );
    renderChat();
    await waitFor(
      () =>
        expect(screen.getAllByRole("button", { name: /retry/i })).toHaveLength(
          2,
        ),
      { timeout: 5000 },
    );
    const [first, second] = screen.getAllByRole("button", { name: /retry/i });

    first.click();
    second.click();

    await screen.findByText(ANSWER, undefined, { timeout: 5000 });
    expect(posts).toEqual([FOLLOW_UP]);
    expect(queued).toEqual([]);
  }, 15_000);

  it("runs nothing again when another turn read the follow-up before the Retry", async () => {
    const bodies: StreamBody[] = [];
    server.use(
      http.post(STREAM_URL, async ({ request }) => {
        bodies.push((await request.json()) as StreamBody);
        // The server finds pm-1 already read by another tab's turn, so it
        // answers "queued" and runs nothing.
        return streamSseResponse(
          [
            { type: "start", messageId: "already-read" },
            { type: "start-step" },
            { type: "data-follow-up-queued", data: {} },
            { type: "finish-step" },
            { type: "finish" },
          ],
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
    // Another tab's turn read the queue and answered it meanwhile.
    buffer.messages = [];
    store("user", FOLLOW_UP);
    store("assistant", ANSWER);

    await userEvent.setup().click(retry);

    await screen.findByText(ANSWER, undefined, { timeout: 5000 });
    expect(bodies).toHaveLength(1);
    expect(bodies[0].resend_pending_id).toBe("pm-1");
    await waitFor(() =>
      expect(screen.queryByText("Didn't go through")).toBeNull(),
    );
    expect(screen.getAllByText(FOLLOW_UP)).toHaveLength(1);
  }, 15_000);

  it("keeps the chip when the expert can no longer be messaged", async () => {
    const expertId = "3f8b0f7e-9f30-4a3b-a6a1-0000000000aa";
    const posts: string[] = [];
    server.use(
      http.get("*/api/experts/identities", () =>
        HttpResponse.json([
          {
            id: expertId,
            name: "Maya",
            avatar_url: null,
            role: "Operations expert",
            is_archived: true,
          },
        ]),
      ),
      http.post(STREAM_URL, ({ request }) => {
        posts.push("sent");
        return streamSseResponse([], { abortSignal: request.signal });
      }),
    );
    renderHost({
      sessionOverride: { messages: stored.messages, expert_id: expertId },
      pendingMessages: buffer.messages,
      pendingHeldReason: "error",
    });
    const retry = await screen.findByRole(
      "button",
      { name: /retry/i },
      { timeout: 5000 },
    );

    await userEvent.setup().click(retry);

    await new Promise((resolve) => setTimeout(resolve, 300));
    expect(posts).toEqual([]);
    expect(screen.getByText("Didn't go through")).toBeDefined();
  });

  it("goes out once on Retry, by its id, and gets its answer", async () => {
    const posts: string[] = [];
    const bodies: StreamBody[] = [];
    server.use(
      http.post(STREAM_URL, async ({ request }) => {
        const body = (await request.json()) as StreamBody;
        bodies.push(body);
        const message = body.message ?? "";
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
    // The server takes that very message off the queue; a second click or a
    // retransmit carries the same key.
    expect(bodies[0]).toMatchObject({
      resend_pending_id: "pm-1",
      message_id: "resend-pm-1",
    });
    await waitFor(() =>
      expect(screen.queryByText("Didn't go through")).toBeNull(),
    );
    expect(screen.getAllByText(FOLLOW_UP)).toHaveLength(1);
  }, 15_000);
});

describe.each([
  ["AI SDK", false],
  ["stream runtime", true],
] as const)("a Retry the server refuses, on the %s stack", (_, runtime) => {
  beforeEach(() => {
    streamPath.runtime = runtime;
  });

  it.each([
    [429, "You've reached your usage limit. Try again later."],
    [402, "Choose a plan to keep going."],
  ])(
    "stays a held chip, not also composer text, when the Retry is refused (%i)",
    async (status, detail) => {
      server.use(
        // Refused before the server took the follow-up: it still waits there.
        http.post(STREAM_URL, () => HttpResponse.json({ detail }, { status })),
      );
      renderChat();
      const retry = await screen.findByRole(
        "button",
        { name: /retry/i },
        { timeout: 5000 },
      );

      await userEvent.setup().click(retry);

      expect(
        await screen.findByText("Didn't go through", undefined, {
          timeout: 5000,
        }),
      ).toBeDefined();
      await new Promise((resolve) => setTimeout(resolve, 300));
      expect(composer().value).toBe("");
      expect(screen.getAllByText(FOLLOW_UP)).toHaveLength(1);
    },
    15_000,
  );
});

describe("a send the server queued behind a turn this tab did not see", () => {
  it("shows once, as a Queued chip, not also as a sent bubble no row backs", async () => {
    buffer.messages = [];
    renderHost({ sessionOverride: { messages: stored.messages } });
    server.use(
      http.post(STREAM_URL, ({ request }) => {
        // Another turn is running: the server buffers the send and says so.
        buffer.messages = [FOLLOW_UP];
        return streamSseResponse(
          [
            { type: "start", messageId: "queued-follow-up" },
            { type: "start-step" },
            { type: "data-follow-up-queued", data: {} },
            { type: "finish-step" },
            { type: "finish" },
          ],
          { abortSignal: request.signal },
        );
      }),
      http.get(PENDING_URL, () =>
        HttpResponse.json({
          messages: buffer.messages,
          count: buffer.messages.length,
          held_reason: null,
        }),
      ),
    );

    await typeAndSend(FOLLOW_UP);

    await screen.findByText("Queued", undefined, { timeout: 5000 });
    await waitFor(() => expect(screen.getAllByText(FOLLOW_UP)).toHaveLength(1));
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
        items: buffer.messages.map((content, index) => ({
          id: `pm-${index + 1}`,
          content,
        })),
        held_reason: buffer.messages.length > 0 ? "error" : null,
      }),
    ),
  );
}

function composer() {
  return document.querySelector<HTMLTextAreaElement>(
    "#chat-input-session, #chat-input",
  )!;
}

interface StreamBody {
  message?: string;
  message_id?: string;
  resend_pending_id?: string | null;
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
