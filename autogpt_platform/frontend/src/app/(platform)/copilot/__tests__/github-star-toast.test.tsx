import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  createStreamHold,
  streamSseResponse,
  type StreamHold,
} from "@/tests/integrations/copilot-sse";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  renderHost,
  resetChatRuntimes,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

// A self-hosted install asks for a GitHub star once, after the fifth prompt;
// the hosted dev and prod sites never ask.

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

const STAR_TOAST_TITLE = "Enjoying AutoGPT?";
const REPO_URL = "https://github.com/Significant-Gravitas/AutoGPT";

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
    created_at: new Date(Date.UTC(2026, 4, 13, 0, sequence)).toISOString(),
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

interface HeldTurn {
  turn: number;
  hold: StreamHold;
}

function answeringStreamHandler(held?: HeldTurn) {
  let turn = 0;
  return http.post(
    `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
    async ({ request }) => {
      const body = (await request.json()) as { message?: string };
      turn += 1;
      const answer = `Answer ${turn}`;
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
        {
          abortSignal: request.signal,
          hold: held?.turn === turn ? held.hold : undefined,
        },
      );
    },
  );
}

// The backend takes a follow-up typed while a turn is still answering.
function pendingFollowUpHandler() {
  return http.post(
    `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/messages/pending`,
    () =>
      HttpResponse.json({
        buffer_length: 1,
        max_buffer_length: 10,
        turn_in_flight: true,
      }),
  );
}

function renderChat(held?: HeldTurn) {
  server.use(answeringStreamHandler(held), pendingFollowUpHandler());
  renderHost();
  // Registered after renderHost so the stored rows win over the empty session.
  server.use(storedSessionHandler());
  render(<Toaster />);
}

async function sendPrompt(turn: number) {
  await typeAndSend(`Prompt ${turn}`);
  await screen.findByText(`Answer ${turn}`, undefined, { timeout: 5000 });
  await waitForTurnToSettle();
}

async function waitForTurnToSettle() {
  await waitFor(
    () => expect(screen.queryByRole("button", { name: "Stop" })).toBeNull(),
    { timeout: 5000 },
  );
}

async function queueFollowUp(text: string) {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText(/chat message input/i), text);
  await user.click(screen.getByRole("button", { name: "Queue message" }));
}

// A browser opens the link in a new tab and leaves this page alone; happy-dom
// would load it into this window, so stop that once the app's handlers ran.
async function clickWithoutNavigating(link: HTMLElement) {
  function stayHere(event: MouseEvent) {
    event.preventDefault();
  }
  document.addEventListener("click", stayHere);
  try {
    await userEvent.click(link);
  } finally {
    document.removeEventListener("click", stayHere);
  }
}

beforeEach(() => {
  resetChatRuntimes();
  stored.messages = [];
  window.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllEnvs();
  resetChatRuntimes();
  server.resetHandlers();
});

describe("GitHub star toast", () => {
  it("asks once, after the fifth prompt, on a self-hosted install", async () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
    renderChat();

    for (let turn = 1; turn <= 4; turn++) await sendPrompt(turn);
    expect(screen.queryByText(STAR_TOAST_TITLE)).toBeNull();

    await sendPrompt(5);
    expect(await screen.findByText(STAR_TOAST_TITLE)).toBeDefined();

    const star = screen.getByRole("link", { name: "Star on GitHub" });
    expect(star.getAttribute("href")).toBe(REPO_URL);
    expect(star.getAttribute("target")).toBe("_blank");
    await clickWithoutNavigating(star);
    await waitFor(() =>
      expect(screen.queryByText(STAR_TOAST_TITLE)).toBeNull(),
    );

    await sendPrompt(6);
    expect(screen.queryByText(STAR_TOAST_TITLE)).toBeNull();
  }, 60_000);

  it("counts a follow-up queued while Otto is still answering", async () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
    const { hold, release } = createStreamHold(1);
    renderChat({ turn: 4, hold });

    for (let turn = 1; turn <= 3; turn++) await sendPrompt(turn);
    await typeAndSend("Prompt 4");
    await screen.findByRole("button", { name: "Stop" });
    await queueFollowUp("And one more thing");
    release();
    await screen.findByText("Answer 4", undefined, { timeout: 5000 });
    await waitForTurnToSettle();

    // Three answered prompts, the fourth and its follow-up make five.
    expect(await screen.findByText(STAR_TOAST_TITLE)).toBeDefined();
  }, 60_000);

  it.each(["dev", "prod"])(
    "never asks on the hosted %s site",
    async (appEnv) => {
      vi.stubEnv("NEXT_PUBLIC_APP_ENV", appEnv);
      renderChat();

      for (let turn = 1; turn <= 5; turn++) await sendPrompt(turn);
      expect(screen.queryByText(STAR_TOAST_TITLE)).toBeNull();
    },
    60_000,
  );
});
