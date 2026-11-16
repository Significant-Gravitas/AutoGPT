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
import type { UIMessageChunk } from "ai";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  clickStop,
  renderHost,
  resetChatRuntimes,
  STREAM_PATHS,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

// A self-hosted install asks for a GitHub star once Otto has answered five
// prompts; a failed or stopped answer doesn't count, and the hosted dev and
// prod sites never ask.

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
      flag === actual.Flag.COPILOT_STREAM_RUNTIME && streamPath.runtime,
  };
});

const STAR_TOAST_TITLE = "Enjoying AutoGPT?";
const REPO_URL = "https://github.com/Significant-Gravitas/AutoGPT";
const FAILURE = "The model provider stopped responding.";

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

interface Turns {
  /** Parks this turn's stream until the test releases it. */
  held?: { turn: number; hold: StreamHold };
  /** This turn starts answering, then fails the way a provider error does. */
  failedTurn?: number;
  /** This turn's stream ends before its finish, and a resume finds it gone. */
  cutTurn?: number;
}

function answeringStreamHandler({ held, failedTurn, cutTurn }: Turns) {
  let turn = 0;
  return http.post(
    `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
    async ({ request }) => {
      const body = (await request.json()) as { message?: string };
      turn += 1;
      const answer = `Answer ${turn}`;
      const failed = turn === failedTurn;
      const cut = turn === cutTurn;
      stored.messages = [
        ...stored.messages,
        sessionMessage(turn * 2 - 1, "user", body.message ?? ""),
        ...(failed || cut
          ? []
          : [sessionMessage(turn * 2, "assistant", answer)]),
      ];
      const chunks: UIMessageChunk[] = assistantTextChunks(answer, {
        messageId: `answer-${turn}`,
        textPartId: `answer-${turn}-text`,
      });
      if (failed) chunks.splice(-2, 2, { type: "error", errorText: FAILURE });
      // The answer's text arrives whole, but the turn's finish never does.
      if (cut) chunks.splice(-2, 2);
      return streamSseResponse(chunks, {
        abortSignal: request.signal,
        hold: held?.turn === turn ? held.hold : undefined,
      });
    },
  );
}

// A resume of a turn the server is no longer running.
function goneStreamHandler() {
  return http.get(
    `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
    () => new HttpResponse(null, { status: 204 }),
  );
}

function renderChat(turns: Turns = {}) {
  server.use(answeringStreamHandler(turns), goneStreamHandler());
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

describe.each(STREAM_PATHS)("GitHub star toast on the %s path", (path) => {
  beforeEach(() => {
    streamPath.runtime = path === "stream runtime";
  });

  it("asks once, after the fifth answer, on a self-hosted install", async () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
    renderChat();

    for (let turn = 1; turn <= 4; turn++) await sendPrompt(turn);
    expect(screen.queryByText(STAR_TOAST_TITLE)).toBeNull();

    await sendPrompt(5);
    expect(await screen.findByText(STAR_TOAST_TITLE)).toBeDefined();

    const star = screen.getByRole("link", {
      name: "Star on GitHub (opens in a new tab)",
    });
    expect(star.getAttribute("href")).toBe(REPO_URL);
    expect(star.getAttribute("target")).toBe("_blank");
    await clickWithoutNavigating(star);
    await waitFor(() =>
      expect(screen.queryByText(STAR_TOAST_TITLE)).toBeNull(),
    );

    await sendPrompt(6);
    expect(screen.queryByText(STAR_TOAST_TITLE)).toBeNull();
  }, 60_000);

  it("doesn't ask beside a failed answer, and asks after the next one", async () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
    renderChat({ failedTurn: 5 });

    for (let turn = 1; turn <= 4; turn++) await sendPrompt(turn);
    await typeAndSend("Prompt 5");
    expect(
      await screen.findByText(FAILURE, { exact: false }, { timeout: 5000 }),
    ).toBeDefined();
    await waitForTurnToSettle();
    expect(screen.queryByText(STAR_TOAST_TITLE)).toBeNull();

    await sendPrompt(6);
    expect(await screen.findByText(STAR_TOAST_TITLE)).toBeDefined();
  }, 60_000);

  it("doesn't count an answer the user stopped", async () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
    const { hold, release } = createStreamHold(1);
    renderChat({ held: { turn: 5, hold } });

    try {
      for (let turn = 1; turn <= 4; turn++) await sendPrompt(turn);
      await typeAndSend("Prompt 5");
      await clickStop();
      await waitForTurnToSettle();
      expect(screen.queryByText(STAR_TOAST_TITLE)).toBeNull();
    } finally {
      release();
    }
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

// Only the turn runtime ends a turn whose stream went away (`expire`).
describe("GitHub star toast on the stream runtime path", () => {
  beforeEach(() => {
    streamPath.runtime = true;
  });

  it("doesn't count an answer whose finish never arrived", async () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
    renderChat({ cutTurn: 5 });

    for (let turn = 1; turn <= 4; turn++) await sendPrompt(turn);
    await sendPrompt(5);
    expect(screen.queryByText(STAR_TOAST_TITLE)).toBeNull();

    await sendPrompt(6);
    expect(await screen.findByText(STAR_TOAST_TITLE)).toBeDefined();
  }, 60_000);
});
