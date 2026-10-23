import {
  getGetV2ListChatTransportsMockHandler200,
  getPostV2CreateSessionMockHandler200,
  getPostV2CreateSessionResponseMock200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  copilotResumeHandler,
  copilotStreamErrorHandler,
  copilotStreamHandler,
  createStreamHold,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { cleanup, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UIMessageChunk } from "ai";
import { delay, http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { SETTLE_GRACE_MS } from "../helpers/replyTiming";
import { useCopilotUIStore } from "../store";
import {
  clickStop,
  copilotStreamSequenceHandler,
  renderHost,
  resetChatRuntimes,
  sessionHandler,
  STREAM_PATHS,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
  waitForInputReady,
} from "./sse-helpers";

// Each reply is reported as one span; the fake keeps what the page asked for.
interface CapturedSpan {
  options: {
    name?: string;
    op?: string;
    forceTransaction?: boolean;
    startTime?: number;
    attributes?: Record<string, unknown>;
  };
  ended: boolean;
  endTime?: number;
}

const sentry = vi.hoisted(() => ({ spans: [] as CapturedSpan[] }));

vi.mock("@sentry/nextjs", async (importActual) => {
  const actual = await importActual<typeof import("@sentry/nextjs")>();
  // Most of the SDK is re-exported from @sentry/node and is not an own key
  // of this namespace, so a spread would drop it; inherit it instead.
  return Object.assign(Object.create(actual), {
    startInactiveSpan(options: CapturedSpan["options"]) {
      const span: CapturedSpan = { options, ended: false };
      sentry.spans.push(span);
      return {
        end(endTime?: number) {
          span.ended = true;
          span.endTime = endTime;
        },
      };
    },
  });
});

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

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT",
    COPILOT_STREAM_RUNTIME: "copilot-stream-runtime",
  },
  useGetFlag: (flag: string) =>
    flag === "copilot-stream-runtime" ? streamPath.runtime : false,
  useFlagStatus: () => ({ enabled: false, ready: true }),
}));

const STREAM_CLIENT = {
  "AI SDK": "ai-sdk",
  "stream runtime": "stream-runtime",
} as const;

// Index 5 of each is its text-start: a hold there keeps the earlier output
// on screen while the answer has not started.
const REASONING_THEN_TEXT: UIMessageChunk[] = [
  { type: "start", messageId: "msg-1" },
  { type: "start-step" },
  { type: "reasoning-start", id: "r1" },
  { type: "reasoning-delta", id: "r1", delta: "Weighing the options." },
  { type: "reasoning-end", id: "r1" },
  { type: "text-start", id: "t1" },
  { type: "text-delta", id: "t1", delta: "Here is the answer." },
  { type: "text-end", id: "t1" },
  { type: "finish-step" },
  { type: "finish" },
];

const TOOL_CALL: UIMessageChunk[] = [
  { type: "tool-input-start", toolCallId: "call-1", toolName: "web_search" },
  {
    type: "tool-input-available",
    toolCallId: "call-1",
    toolName: "web_search",
    input: { query: "weather" },
  },
  {
    type: "tool-output-available",
    toolCallId: "call-1",
    output: { result: "sunny" },
  },
];

// The server moved the chat to another engine; it starts the next turn itself.
const ENGINE_SWITCH: UIMessageChunk = {
  type: "data-mode-changed",
  data: { mode: "extended_thinking" },
};

const TOOL_THEN_TEXT: UIMessageChunk[] = [
  { type: "start", messageId: "msg-1" },
  { type: "start-step" },
  ...TOOL_CALL,
  { type: "text-start", id: "t1" },
  { type: "text-delta", id: "t1", delta: "Here is the answer." },
  { type: "text-end", id: "t1" },
  { type: "finish-step" },
  { type: "finish" },
];
const TEXT_START = 5;
// Long enough to tell a measurement of the wait from one taken without it.
const HELD_MS = 150;

function replySpans() {
  return sentry.spans.filter((span) => span.options.op === "copilot.reply");
}

/** Spans that were started and ended: one that is never ended is never sent. */
async function recordedReplies(count: number) {
  await waitFor(
    () => {
      expect(replySpans()).toHaveLength(count);
      expect(replySpans().every((span) => span.ended)).toBe(true);
    },
    { timeout: 5000 },
  );
  return replySpans();
}

async function recordedReply() {
  const [span] = await recordedReplies(1);
  return span;
}

function attributesOf(span: CapturedSpan) {
  return span.options.attributes ?? {};
}

function msOf(span: CapturedSpan, name: "first_output_ms" | "first_text_ms") {
  return attributesOf(span)[`copilot.reply.${name}`] as number;
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Type and submit; returns the clock just before and after the click. */
async function sendAndClock(text: string) {
  await waitForInputReady();
  const user = userEvent.setup();
  await user.type(screen.getByLabelText(/chat message input/i), text);
  const submit = screen.getByRole("button", { name: /submit/i });
  const before = performance.now();
  await user.click(submit);
  return { before, after: performance.now() };
}

async function waitForTurnOver() {
  await waitFor(
    () => expect(screen.queryByRole("button", { name: /stop/i })).toBeNull(),
    { timeout: 5000 },
  );
}

beforeEach(() => {
  resetChatRuntimes();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
  sentry.spans = [];
});

afterEach(async () => {
  // A reply still followed when the chat unmounts is given up and reported a
  // few ticks later; let it land here rather than in the next test.
  cleanup();
  await sleep(20);
  resetChatRuntimes();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
});

describe.each(STREAM_PATHS)("on the %s path", (path) => {
  beforeEach(() => {
    streamPath.runtime = path === "stream runtime";
  });

  describe("AutoPilot reply timing", () => {
    it("records the time from the send to the first answer text", async () => {
      const { hold, release } = createStreamHold(3);
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: assistantTextChunks("The answer is 42."),
          hold,
        }),
      );

      renderHost();
      const sent = await sendAndClock("what is the answer?");
      await screen.findByRole("button", { name: /stop/i });
      await sleep(HELD_MS);
      expect(replySpans()).toHaveLength(0);

      const releasedAt = performance.now();
      release();
      expect(await screen.findByText("The answer is 42.")).toBeDefined();
      const span = await recordedReply();
      const recordedBy = performance.now();

      expect(span.options).toMatchObject({
        name: "copilot reply",
        op: "copilot.reply",
        forceTransaction: true,
      });
      expect(attributesOf(span)).toMatchObject({
        "copilot.reply.outcome": "answered",
        "copilot.reply.first_output_kind": "text",
        "copilot.reply.new_chat": false,
        "copilot.reply.expert": false,
        "copilot.reply.stream_client": STREAM_CLIENT[path],
      });
      const firstText = msOf(span, "first_text_ms");
      expect(msOf(span, "first_output_ms")).toBe(firstText);
      // From the click, not from the stream or the render: it spans the hold.
      expect(firstText).toBeGreaterThanOrEqual(releasedAt - sent.after - 1);
      expect(firstText).toBeLessThanOrEqual(recordedBy - sent.before + 1);
      // The span starts at the send and ends at the text.
      const startTime = span.options.startTime ?? 0;
      expect(startTime).toBeGreaterThanOrEqual(
        performance.timeOrigin + sent.before,
      );
      expect(startTime).toBeLessThanOrEqual(
        performance.timeOrigin + sent.after,
      );
      expect(span.ended).toBe(true);
      expect((span.endTime ?? 0) - startTime).toBeCloseTo(firstText, -1);
    });

    it("times every send that starts a turn", async () => {
      server.use(
        copilotStreamSequenceHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunksPerTurn: [
            assistantTextChunks("First reply.", { messageId: "msg-1" }),
            assistantTextChunks("Second reply.", { messageId: "msg-2" }),
          ],
        }),
      );

      renderHost();
      await typeAndSend("first message");
      await screen.findByText("First reply.", undefined, { timeout: 5000 });
      await recordedReplies(1);
      await waitForTurnOver();

      await typeAndSend("second message");
      await screen.findByText("Second reply.", undefined, { timeout: 5000 });
      const spans = await recordedReplies(2);

      for (const span of spans) {
        expect(attributesOf(span)).toMatchObject({
          "copilot.reply.outcome": "answered",
          "copilot.reply.first_output_kind": "text",
        });
      }
    });

    it.each([
      ["reasoning", REASONING_THEN_TEXT, "Weighing the options."],
      ["tool", TOOL_THEN_TEXT, /searched the web/i],
    ] as const)(
      "records a %s shown before the text as the first output",
      async (kind, chunks, shownOutput) => {
        const { hold, release } = createStreamHold(TEXT_START);
        server.use(
          copilotStreamHandler({
            baseUrl: TEST_BACKEND_BASE_URL,
            sessionId: TEST_SESSION_ID,
            chunks: [...chunks],
            hold,
          }),
        );

        renderHost();
        await typeAndSend("how is the weather?");
        expect(
          await screen.findAllByText(shownOutput, undefined, { timeout: 5000 }),
        ).not.toHaveLength(0);
        await sleep(HELD_MS);
        expect(replySpans()).toHaveLength(0);

        release();
        await screen.findByText("Here is the answer.", undefined, {
          timeout: 5000,
        });
        const span = await recordedReply();

        expect(attributesOf(span)).toMatchObject({
          "copilot.reply.outcome": "answered",
          "copilot.reply.first_output_kind": kind,
        });
        expect(
          msOf(span, "first_text_ms") - msOf(span, "first_output_ms"),
        ).toBeGreaterThanOrEqual(HELD_MS - 1);
      },
    );

    it("ends a turn that shows no answer text as no_text", async () => {
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: [
            { type: "start", messageId: "msg-1" },
            { type: "start-step" },
            ...TOOL_CALL,
            { type: "finish-step" },
            { type: "finish" },
          ],
        }),
      );

      renderHost();
      await typeAndSend("look it up");
      const attributes = attributesOf(await recordedReply());

      expect(attributes).toMatchObject({
        "copilot.reply.outcome": "no_text",
        "copilot.reply.first_output_kind": "tool",
      });
      expect(attributes).not.toHaveProperty("copilot.reply.first_text_ms");
    });

    it("times a reply to the text of the turn the server starts after it", async () => {
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: [
            { type: "start", messageId: "msg-1" },
            { type: "start-step" },
            ...TOOL_CALL,
            ENGINE_SWITCH,
            { type: "finish-step" },
            { type: "finish" },
          ],
        }),
        copilotResumeHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: assistantTextChunks("Your agent is ready.", {
            messageId: "msg-2",
          }),
        }),
      );

      renderHost();
      await typeAndSend("build me an agent");
      await screen.findAllByText(/searched the web/i, undefined, {
        timeout: 5000,
      });
      // Dispatched after the turn ends, later than a turn that ended without
      // text is given to show some.
      await sleep(SETTLE_GRACE_MS + HELD_MS);
      server.use(
        sessionHandler({
          chat_status: "running",
          active_stream: {
            turn_id: "continuation-turn",
            last_message_id: "msg-1",
            started_at: "2026-05-13T00:00:00Z",
          },
        }),
      );
      await screen.findByText("Your agent is ready.", undefined, {
        timeout: 5000,
      });
      server.use(sessionHandler());
      const span = await recordedReply();

      expect(attributesOf(span)).toMatchObject({
        "copilot.reply.outcome": "answered",
        "copilot.reply.first_output_kind": "tool",
      });
      expect(
        msOf(span, "first_text_ms") - msOf(span, "first_output_ms"),
      ).toBeGreaterThanOrEqual(SETTLE_GRACE_MS + HELD_MS);
    }, 15_000);

    it("does not time a follow-up queued behind the running turn", async () => {
      const { hold, release } = createStreamHold(3);
      const queued = vi.fn(() =>
        HttpResponse.json({
          buffer_length: 1,
          max_buffer_length: 10,
          turn_in_flight: true,
        }),
      );
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: assistantTextChunks("First answer."),
          hold,
        }),
        http.post(
          `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/messages/pending`,
          queued,
        ),
      );

      renderHost();
      await typeAndSend("first question");
      await screen.findByRole("button", { name: /stop/i });
      const user = userEvent.setup();
      await user.type(
        screen.getByLabelText(/chat message input/i),
        "and a follow-up",
      );
      await user.click(screen.getByRole("button", { name: /queue message/i }));
      await waitFor(() => expect(queued).toHaveBeenCalledTimes(1));

      release();
      await screen.findByText("First answer.", undefined, { timeout: 5000 });
      await waitForTurnOver();
      await sleep(HELD_MS);

      const [span] = await recordedReplies(1);
      expect(attributesOf(span)["copilot.reply.outcome"]).toBe("answered");
    });

    it("ends a reply that errors before its text as abandoned", async () => {
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: [
            { type: "start", messageId: "msg-1" },
            { type: "start-step" },
            { type: "error", errorText: "Backend went sideways." },
          ],
        }),
      );

      renderHost();
      await typeAndSend("hi");
      await screen.findByText(/backend went sideways\./i, undefined, {
        timeout: 5000,
      });
      const span = await recordedReply();

      expect(attributesOf(span)["copilot.reply.outcome"]).toBe("abandoned");
      expect(attributesOf(span)).not.toHaveProperty(
        "copilot.reply.first_output_ms",
      );
      expect(attributesOf(span)).not.toHaveProperty(
        "copilot.reply.first_text_ms",
      );
    });

    it("ends a send the server fails as abandoned", async () => {
      server.use(
        copilotStreamErrorHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          status: 500,
          body: "kaboom",
        }),
      );

      renderHost();
      await typeAndSend("hi");
      await screen.findByText(/kaboom/i, undefined, { timeout: 5000 });
      const span = await recordedReply();

      expect(attributesOf(span)["copilot.reply.outcome"]).toBe("abandoned");
    });

    it("records nothing for a send refused back into the composer", async () => {
      server.use(
        copilotStreamErrorHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          status: 429,
          body: { detail: "You've reached your usage limit. Try again later." },
        }),
      );

      renderHost();
      await typeAndSend("over the limit");
      await screen.findByText(/We couldn’t check your usage/i, undefined, {
        timeout: 5000,
      });
      await waitFor(() =>
        expect(
          (screen.getByLabelText(/chat message input/i) as HTMLTextAreaElement)
            .value,
        ).toBe("over the limit"),
      );
      // Past the AI SDK's post-turn probe, where a settled turn would end.
      await sleep(1000);

      expect(replySpans()).toHaveLength(0);
    });

    it("ends a reply the user stops before its text as abandoned", async () => {
      const { hold } = createStreamHold(3);
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: assistantTextChunks("Never shown."),
          hold,
        }),
      );

      renderHost();
      await typeAndSend("hi");
      await clickStop();
      const span = await recordedReply();

      expect(attributesOf(span)["copilot.reply.outcome"]).toBe("abandoned");
      expect(attributesOf(span)).not.toHaveProperty(
        "copilot.reply.first_text_ms",
      );
      expect(screen.queryByText("Never shown.")).toBeNull();
    });

    it("ends a reply left for another chat before its text as abandoned", async () => {
      const { hold } = createStreamHold(TEXT_START);
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: [...REASONING_THEN_TEXT],
          hold,
        }),
      );

      // CopilotPage keys the host by session, so switching chats unmounts it.
      const { unmount } = renderHost();
      await typeAndSend("think about it");
      await screen.findByText("Weighing the options.", undefined, {
        timeout: 5000,
      });
      expect(replySpans()).toHaveLength(0);
      unmount();
      const span = await recordedReply();

      expect(attributesOf(span)).toMatchObject({
        "copilot.reply.outcome": "abandoned",
        "copilot.reply.first_output_kind": "reasoning",
      });
      expect(attributesOf(span)).not.toHaveProperty(
        "copilot.reply.first_text_ms",
      );
    });

    // Strict Mode, as the dev server runs: the new host's mount effects, one
    // of which dispatches the first send, run twice.
    it.each([
      ["", false],
      [" under Strict Mode", true],
    ])(
      "times a new chat's first send across the session it creates%s",
      async (_, strictMode) => {
        // Session creation is part of what the user waits for.
        const creatingMs = 200;
        server.use(
          getGetV2ListChatTransportsMockHandler200({
            transports: [
              {
                auth_provider: "platform",
                credential_id: null,
                label: "AutoGPT Platform",
                available: true,
                default: true,
              },
            ],
          }),
          getPostV2CreateSessionMockHandler200(async () => {
            await delay(creatingMs);
            return getPostV2CreateSessionResponseMock200({
              id: TEST_SESSION_ID,
            });
          }),
          copilotStreamHandler({
            baseUrl: TEST_BACKEND_BASE_URL,
            sessionId: TEST_SESSION_ID,
            chunks: assistantTextChunks("Welcome aboard."),
          }),
        );

        renderHost({ newChat: true, strictMode });
        await typeAndSend("start something new");
        await screen.findByText("Welcome aboard.", undefined, {
          timeout: 5000,
        });
        const span = await recordedReply();

        expect(attributesOf(span)).toMatchObject({
          "copilot.reply.outcome": "answered",
          "copilot.reply.new_chat": true,
          "copilot.reply.stream_client": STREAM_CLIENT[path],
        });
        expect(msOf(span, "first_text_ms")).toBeGreaterThanOrEqual(creatingMs);
      },
    );
  });
});

function storedMessage(
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

// Only the AI SDK client drops a repeat of a message still waiting for its
// answer, so this runs on that path alone: CI rejects a report with a skipped
// test.
describe("on the AI SDK path", () => {
  beforeEach(() => {
    streamPath.runtime = false;
  });

  it("records nothing for a send suppressed as a duplicate", async () => {
    // Reloaded while the second "same again" waits for its answer: the tab
    // still remembers sending it, so the same text again is a double press.
    // The answered copy above it must not be taken for this send.
    useCopilotStreamStore.getState().updateCoord(TEST_SESSION_ID, {
      lastSubmittedMessageText: "same again",
    });
    const posts = vi.fn();
    server.use(
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
        ({ request }) => {
          posts();
          return streamSseResponse(assistantTextChunks("Another reply."), {
            abortSignal: request.signal,
          });
        },
      ),
    );

    renderHost({
      sessionOverride: {
        messages: [
          storedMessage(1, "user", "same again"),
          storedMessage(2, "assistant", "Only reply."),
          storedMessage(3, "user", "same again"),
        ],
      },
    });
    await screen.findByText("Only reply.", undefined, { timeout: 5000 });
    await typeAndSend("same again");
    await sleep(SETTLE_GRACE_MS + HELD_MS);

    expect(screen.getByLabelText(/chat message input/i)).toHaveProperty(
      "value",
      "",
    );
    expect(screen.getAllByText("same again")).toHaveLength(2);
    // Leaving the chat ends the timing too, still unreported.
    cleanup();
    await sleep(20);
    expect(posts).not.toHaveBeenCalled();
    expect(replySpans()).toHaveLength(0);
  });
});
