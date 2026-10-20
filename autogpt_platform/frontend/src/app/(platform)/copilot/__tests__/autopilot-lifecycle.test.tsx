import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  copilotResumeHandler,
  copilotStreamHandler,
  createStreamHold,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  clickStop,
  renderHost,
  resetChatRuntimes,
  sessionHandler,
  STREAM_PATHS,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
  type SessionOverride,
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

const streamPath = vi.hoisted(() => ({ runtime: false }));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT",
    COPILOT_STREAM_RUNTIME: "copilot-stream-runtime",
  },
  useGetFlag: (flag: string) =>
    flag === "copilot-stream-runtime" ? streamPath.runtime : false,
}));

beforeEach(() => {
  resetChatRuntimes();
});

afterEach(() => {
  resetChatRuntimes();
});

describe.each(STREAM_PATHS)("on the %s path", (path) => {
  beforeEach(() => {
    streamPath.runtime = path === "stream runtime";
  });

  it.each([undefined, null])(
    "can send and receive with missing session messages (%s)",
    async (messages) => {
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: assistantTextChunks("Still able to chat."),
        }),
      );
      renderHost({
        sessionResponse: http.get(
          `*/api/chat/sessions/${TEST_SESSION_ID}`,
          () =>
            HttpResponse.json({
              id: TEST_SESSION_ID,
              created_at: "2026-05-13T00:00:00Z",
              updated_at: "2026-05-13T00:00:00Z",
              user_id: "test-user",
              messages,
              active_stream: null,
            }),
        ),
      });

      await typeAndSend("hello");

      expect(
        await screen.findByText("Still able to chat.", undefined, {
          timeout: 5000,
        }),
      ).toBeDefined();
    },
  );

  it.each([null, "expert_run", "scheduled_chat_notice"])(
    "receives a background follow-up without sending again after %s",
    async (noticeKind) => {
      const pending = [
        {
          role: "user",
          content: "Research this in the background.",
          sequence: 0,
        },
        {
          role: "assistant",
          content: null,
          sequence: 1,
          tool_calls: [
            {
              id: "background-call",
              type: "function",
              function: {
                name: "run_sub_session",
                arguments: '{"background":true}',
              },
            },
          ],
        },
        {
          role: "tool",
          tool_call_id: "background-call",
          sequence: 2,
          content: JSON.stringify({
            type: "sub_session_status",
            status: "queued",
            background_task_id: "background-task",
            background_expires_at: new Date(Date.now() + 60_000).toISOString(),
          }),
        },
        {
          role: "assistant",
          content: "I will return with the result.",
          sequence: 3,
        },
      ];
      let sends = 0;
      server.use(
        http.post(
          `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
          () => {
            sends += 1;
            return HttpResponse.error();
          },
        ),
      );
      renderHost({ sessionOverride: { messages: pending } });
      await screen.findByText("I will return with the result.");
      const queued: NonNullable<SessionOverride["messages"]> = [
        ...pending,
        {
          role: "user",
          content: "Internal follow-up prompt",
          sequence: 4,
          metadata: {
            background_task_notice: "background-task",
            background_task_stage: "followup",
            display_content:
              "Background subtask completed. Otto is following up.",
          },
        },
      ];
      if (noticeKind) {
        queued.push({
          role: "assistant",
          content: "An unrelated update.",
          sequence: 5,
          metadata: {
            kind: noticeKind,
            headline: "An unrelated update.",
            body: "A scheduled run was skipped.",
          },
        });
        server.use(sessionHandler({ messages: queued, chat_status: "queued" }));
        await screen.findByText(
          "Background subtask completed. Otto is following up.",
          undefined,
          { timeout: 6000 },
        );
      }
      server.use(
        sessionHandler({
          messages: [
            ...queued,
            {
              role: "assistant",
              content: "The background result is 42.",
              sequence: queued.length,
            },
          ],
        }),
      );
      expect(
        await screen.findByText("The background result is 42.", undefined, {
          timeout: 6000,
        }),
      ).toBeDefined();
      expect(screen.queryByText("Internal follow-up prompt")).toBeNull();
      expect(sends).toBe(0);
    },
    15_000,
  );

  describe("Otto streaming — submit / stop lifecycle", () => {
    it("swaps the submit button to Stop while streaming and back to Submit when done", async () => {
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: assistantTextChunks("Hi."),
          delayMsBetweenChunks: 30,
        }),
      );

      renderHost();
      expect(screen.getByRole("button", { name: /submit/i })).toBeDefined();

      await typeAndSend("hi");

      expect(await screen.findByRole("button", { name: "Stop" })).toBeDefined();

      expect(
        await screen.findByText("Hi.", undefined, { timeout: 5000 }),
      ).toBeDefined();
      await waitFor(() => {
        expect(screen.queryByRole("button", { name: "Stop" })).toBeNull();
      });
    });

    it("ends the stream and shows the manual-stop marker when Stop is clicked mid-stream", async () => {
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: [
            { type: "start", messageId: "msg-1" },
            { type: "start-step" },
            { type: "text-start", id: "t1" },
            { type: "text-delta", id: "t1", delta: "Streaming " },
            { type: "text-delta", id: "t1", delta: "in progress " },
            { type: "text-delta", id: "t1", delta: "should be cut off" },
            { type: "text-end", id: "t1" },
            { type: "finish-step" },
            { type: "finish" },
          ],
          // Setup + first delta emit instantly; stall before the "should be cut
          // off" delta so the stop click has a wide window to land.
          perChunkDelaysMs: [0, 0, 0, 0, 5000, 0, 0, 0, 0],
        }),
      );

      renderHost();
      await typeAndSend("hi");

      // Wait for the first delta so Stop has something live to abort.
      await screen.findByText(/streaming/i, undefined, { timeout: 5000 });

      await clickStop();

      expect(
        await screen.findByText(/task stopped/i, undefined, {
          timeout: 5000,
        }),
      ).toBeDefined();
      // Post-stop chunks never reach the bubble.
      expect(screen.queryByText(/should be cut off/i)).toBeNull();
    });

    // A phone user's second taps came up to 1.5 s after Send, while his sends
    // took up to 1.6 s to name their turn: past Stop's arming, still connecting.
    it("a second tap on Send while the send connects neither stops nor loses it", async () => {
      const { hold, release } = createStreamHold(0);
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: assistantTextChunks("Here is the answer."),
          hold,
        }),
      );
      renderHost();
      const cancels = countCancels();

      await typeAndSend("make some screenshots");
      await sleep(1200);
      await userEvent.setup().click(sendSlot());
      await sleep(200);
      expect(cancels()).toBe(0);

      release();
      expect(
        await screen.findByText("Here is the answer.", undefined, {
          timeout: 5000,
        }),
      ).toBeDefined();
    }, 15_000);

    // The same user's second tap landed 0.15 s after Stop replaced Send.
    it("a Stop that has just replaced Send ignores a tap, then stops once armed", async () => {
      const { hold, release } = createStreamHold(4);
      server.use(
        copilotStreamHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: [
            { type: "start", messageId: "msg-1" },
            { type: "start-step" },
            { type: "text-start", id: "t1" },
            { type: "text-delta", id: "t1", delta: "Working on it " },
            { type: "text-delta", id: "t1", delta: "and done." },
            { type: "text-end", id: "t1" },
            { type: "finish-step" },
            { type: "finish" },
          ],
          hold,
        }),
      );
      renderHost();
      const cancels = countCancels();
      const user = userEvent.setup();

      await typeAndSend("hello");
      const stop = await screen.findByRole("button", { name: /^stop$/i });
      await sleep(150);
      await user.click(stop);
      await sleep(200);
      expect(cancels()).toBe(0);

      await clickStop();
      await waitFor(() => expect(cancels()).toBe(1));
      expect(
        await screen.findByText(/task stopped/i, undefined, { timeout: 5000 }),
      ).toBeDefined();
      release();
    }, 15_000);
  });

  describe("Otto streaming — resume on mount", () => {
    it("issues a GET resume and renders streamed content when the session has an active_stream", async () => {
      server.use(
        copilotResumeHandler({
          baseUrl: TEST_BACKEND_BASE_URL,
          sessionId: TEST_SESSION_ID,
          chunks: assistantTextChunks("Resumed content here."),
        }),
      );

      renderHost({
        sessionOverride: {
          active_stream: {
            turn_id: "turn-1",
            last_message_id: "msg-prev",
            started_at: "2026-05-13T00:00:00Z",
          },
        },
      });

      expect(
        await screen.findByText("Resumed content here.", undefined, {
          timeout: 5000,
        }),
      ).toBeDefined();
    });
  });
});

function countCancels() {
  let count = 0;
  server.use(
    http.post(`*/api/proxy/api/chat/sessions/${TEST_SESSION_ID}/cancel`, () => {
      count += 1;
      return HttpResponse.json({ cancelled: true, reason: null });
    }),
  );
  return () => count;
}

// Whatever button sits where Send was: the composer's last.
function sendSlot() {
  const form = screen.getByLabelText(/chat message input/i).closest("form");
  if (!form) throw new Error("no composer form");
  const buttons = within(form).getAllByRole("button");
  return buttons[buttons.length - 1];
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
