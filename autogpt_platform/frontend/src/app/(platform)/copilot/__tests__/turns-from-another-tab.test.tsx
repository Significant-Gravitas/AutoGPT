import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { installLiveSocket } from "@/tests/integrations/live-socket";
import type { FakeWebSocket } from "@/lib/autogpt-server-api/__tests__/websocket-fake";
import { act, waitFor } from "@testing-library/react";
import { http } from "msw";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useCopilotUIStore } from "../store";
import { getTurnRuntime } from "../stream/turnRuntime";
import { persistedPromptId } from "../stream/turnTail";
import {
  renderHost,
  resetChatRuntimes,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

// The same chat open in two tabs: this page is one of them, and the other
// tab's turns reach it only as rows the session GET returns (#17355).
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
  useAuth: () => ({
    user: { id: USER_ID },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));
vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    COPILOT_STREAM_RUNTIME: "copilot-stream-runtime",
  },
  useGetFlag: (flag: string) => flag === "copilot-stream-runtime",
}));
vi.mock("@/hooks/usePlatformPayment", () => ({
  usePlatformPayment: () => ({ isPaymentEnabled: false, isResolved: true }),
}));

const USER_ID = "test-user";
const STREAM = `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`;
const LIMA = "Reply with exactly one word: lima";
const MIKE = "Reply with exactly one word: mike";
const NOVEMBER = "Reply with exactly one word: november";
// A turn's rows are read half a second after its finish, and a busy runner
// can take seconds more to read and render them.
const SETTLED = { timeout: 10_000 };
// Above two SETTLED waits, so a slow runner fails on the transcript, not
// on the generic test timeout.
const TEST_TIMEOUT = { timeout: 25_000 };

type Row = SessionDetailResponseMessagesItem;

let rows: Row[] = [];

function resetStores() {
  resetChatRuntimes();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
  window.localStorage.clear();
}

beforeEach(() => {
  resetStores();
  rows = [];
  store("user", LIMA);
  store("assistant", "Lima.");
});

afterEach(() => {
  resetStores();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it(
  "a send from a tab that missed the other tab's turn shows that turn above its own, without a reload",
  TEST_TIMEOUT,
  async () => {
    await openChat();

    // The other tab's turn ends while this one is idle; no word of it arrives.
    otherTabSends(MIKE, "Mike.");
    answerSends("November.");
    await typeAndSend(NOVEMBER);

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          `user: ${LIMA}`,
          "assistant: Lima.",
          `user: ${MIKE}`,
          "assistant: Mike.",
          `user: ${NOVEMBER}`,
          "assistant: November.",
        ]),
      SETTLED,
    );
  },
);

it(
  "an idle chat shows a turn the other tab ended once the server says so",
  TEST_TIMEOUT,
  async () => {
    const socket = await openChat();

    otherTabSends(MIKE, "Mike.");
    await turnEnded(socket);

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          `user: ${LIMA}`,
          "assistant: Lima.",
          `user: ${MIKE}`,
          "assistant: Mike.",
        ]),
      SETTLED,
    );
  },
);

it(
  "an idle chat shows a turn the other tab ended while the live connection was down, once it is back",
  TEST_TIMEOUT,
  async () => {
    const live = installLiveSocket();
    const socket = await openChat(live);

    // The turn's end was said while the connection was down.
    await live.drop(socket);
    otherTabSends(MIKE, "Mike.");
    await live.acceptNext();
    await live.caughtUp();

    await waitFor(
      () =>
        expect(transcript()).toEqual([
          `user: ${LIMA}`,
          "assistant: Lima.",
          `user: ${MIKE}`,
          "assistant: Mike.",
        ]),
      SETTLED,
    );
  },
);

it(
  "reads the chat again only for a turn that ended in it",
  TEST_TIMEOUT,
  async () => {
    const socket = await openChat();
    const reread = vi.spyOn(getTurnRuntime(TEST_SESSION_ID), "rereadIfIdle");

    await turnEnded(socket, "another-session");
    expect(reread).not.toHaveBeenCalled();

    await turnEnded(socket);
    expect(reread).toHaveBeenCalledTimes(1);
  },
);

/** Opens this tab on the chat, with the live connection the page keeps. */
async function openChat(live = installLiveSocket()) {
  renderHost({
    sessionResponse: getGetV2GetSessionMockHandler200(() => session()),
  });
  const socket = await live.acceptNext();
  await waitFor(
    () => expect(transcript()).toEqual([`user: ${LIMA}`, "assistant: Lima."]),
    SETTLED,
  );
  return socket;
}

/** Another tab's whole turn, as the session stores it. */
function otherTabSends(prompt: string, answer: string) {
  store("user", prompt);
  store("assistant", answer);
}

/** This tab's sends: the backend stores the prompt and answer, then streams. */
function answerSends(answer: string) {
  server.use(
    http.post(STREAM, async ({ request }) => {
      const body = (await request.json()) as {
        message: string;
        message_id: string;
      };
      store(
        "user",
        body.message,
        persistedPromptId(USER_ID, TEST_SESSION_ID, body.message_id),
      );
      store("assistant", answer);
      return streamSseResponse(assistantTextChunks(answer), {
        abortSignal: request.signal,
      });
    }),
  );
}

/** What the server says to every tab of the user when a turn ends. */
async function turnEnded(socket: FakeWebSocket, sessionId = TEST_SESSION_ID) {
  await act(async () =>
    socket.receive({
      method: "notification",
      data: {
        type: "copilot_completion",
        event: "session_completed",
        session_id: sessionId,
        status: "completed",
      },
    }),
  );
}

function session(): SessionDetailResponse {
  return {
    id: TEST_SESSION_ID,
    created_at: "2026-11-24T00:00:00Z",
    updated_at: "2026-11-24T00:00:00Z",
    user_id: USER_ID,
    chat_status: "idle",
    messages: rows.map((r) => ({ ...r })),
    has_more_messages: false,
    oldest_sequence: null,
    active_stream: null,
    metadata: { dry_run: false, builder_graph_id: null },
    expert_id: null,
    background_pending: false,
  };
}

/** Appends a row to the session, at the next sequence. */
function store(role: "user" | "assistant", content: string, id?: string) {
  const sequence = rows.length;
  rows.push({
    id: id ?? `row-${sequence}`,
    role,
    content,
    tool_call_id: null,
    tool_calls: null,
    sequence,
    duration_ms: null,
    created_at: `2026-11-24T00:00:${String(sequence).padStart(2, "0")}Z`,
    metadata: null,
  });
}

/**
 * Every rendered message, in order, as `role: text`, by the chat's messages
 * each one shows: a bubble also holds its time and its action buttons, and
 * one that shows two messages names both.
 */
function transcript() {
  const texts = rows.map((r) => String(r.content));
  return Array.from(document.querySelectorAll("[data-message-id]")).map(
    (el) => {
      const role = el.classList.contains("is-user") ? "user" : "assistant";
      const shown = el.textContent ?? "";
      const known = texts.filter((text) => shown.includes(text));
      return `${role}: ${known.join(" + ") || shown}`;
    },
  );
}
