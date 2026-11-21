import { useCopilotStream } from "@/app/(platform)/copilot/useCopilotStream";
import { server } from "@/mocks/mock-server";
import {
  createStreamHold,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { UIMessage, UIMessageChunk } from "ai";
import { delay, http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useBuilderChatSend } from "../useBuilderChatSend";

// A follow-up sent in the builder chat while its Stop settles reaches a server
// that refuses to queue it (the stop hold, SECRT-3127). The panel must keep it
// as one queued chip and send it as the next turn once the Stop has settled.

const BASE_URL = "http://localhost:18006";
const SESSION_ID = "builder-stop-session";
const SESSION_URL = `${BASE_URL}/api/chat/sessions/${SESSION_ID}`;
// The generated client reaches these through the app's proxy.
const PROXIED_SESSION_URL = `*/api/chat/sessions/${SESSION_ID}`;
const FIRST = "Build me a weekly report workflow";
const FOLLOW_UP = "Use my Gmail account instead";
const LIVE_PARTIAL = "Adding a Gmail block";
const NO_MESSAGES: UIMessage[] = [];

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => BASE_URL,
    },
  };
});

vi.mock("@/app/(platform)/copilot/helpers", async (importActual) => {
  const actual =
    await importActual<typeof import("@/app/(platform)/copilot/helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

const captureException = vi.hoisted(() => vi.fn());
vi.mock("@sentry/nextjs", async (importActual) => ({
  ...(await importActual<typeof import("@sentry/nextjs")>()),
  captureException,
}));

beforeEach(() => {
  captureException.mockClear();
});

afterEach(() => {
  server.resetHandlers();
});

describe("Builder chat — a follow-up sent while Stop settles", () => {
  it("is refused by the stop hold, shown as one queued chip, and sent as the next turn", async () => {
    const backend = fakeBackend();
    const chat = renderBuilderChat();

    act(() => {
      void chat.result.current.onSend(FIRST);
    });
    await waitFor(
      () => expect(lastText(chat.result.current.messages)).toBe(LIVE_PARTIAL),
      { timeout: 5000 },
    );
    expect(chat.result.current.status).toBe("streaming");

    // Sent in the same moment as the Stop, so the panel still reads streaming
    // and queues it behind the turn being stopped.
    let stopped!: Promise<unknown>;
    act(() => {
      stopped = chat.result.current.stop();
      void chat.result.current.onSend(FOLLOW_UP);
    });

    await waitFor(() => expect(backend.refusedFollowUps).toEqual([FOLLOW_UP]));
    await waitFor(() => expect(backend.turns).toHaveLength(2), {
      timeout: 5000,
    });
    expect(backend.turns[1]).toContain(FOLLOW_UP);
    await act(async () => {
      expect(await stopped).toEqual([]);
    });
    expect(backend.cancels).toBe(1);
    await waitFor(() => expect(chat.result.current.queuedMessages).toEqual([]));
    // The chip can last a single render before the settled Stop sends it.
    expect(chat.chipsShown).toContainEqual([FOLLOW_UP]);
    expect(chat.chipsShown.filter((chips) => chips.length > 1)).toEqual([]);
    expect(captureException).not.toHaveBeenCalled();
  }, 20_000);
});

function renderBuilderChat() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }
  const chipsShown: string[][] = [];
  const view = renderHook(() => useBuilderChat(chipsShown), {
    wrapper: Wrapper,
  });
  return { ...view, chipsShown };
}

// Wired as `useBuilderChatPanel` wires them.
function useBuilderChat(chipsShown: string[][]) {
  const stream = useCopilotStream({
    sessionId: SESSION_ID,
    hydratedMessages: NO_MESSAGES,
    hasActiveStream: false,
    refetchSession,
    copilotModel: undefined,
  });
  const { onSend, queuedMessages } = useBuilderChatSend({
    sessionId: SESSION_ID,
    status: stream.status,
    isReconnecting: stream.isReconnecting,
    isFinishProbing: stream.isFinishProbing,
    messages: stream.messages,
    setMessages: stream.setMessages,
    sendMessage: stream.sendMessage,
  });
  chipsShown.push(queuedMessages);
  return {
    messages: stream.messages,
    status: stream.status,
    stop: stream.stop,
    onSend,
    queuedMessages,
  };
}

// The server as the stop hold makes it: once the cancel has started, the
// follow-up gate refuses to queue behind the stopped turn, and `/stream`
// starts the next one.
function fakeBackend() {
  const backend = {
    turns: [] as string[],
    refusedFollowUps: [] as string[],
    cancels: 0,
    stopHeld: false,
  };
  const { hold } = createStreamHold(4);
  server.use(
    http.post(`${SESSION_URL}/stream`, async ({ request }) => {
      backend.turns.push(await request.clone().text());
      const isFirstTurn = backend.turns.length === 1;
      return streamSseResponse(
        isFirstTurn ? stoppableTurnChunks() : answerChunks(),
        isFirstTurn
          ? { hold, abortSignal: request.signal }
          : { abortSignal: request.signal },
      );
    }),
    http.post(`${SESSION_URL}/messages/pending`, async ({ request }) => {
      const { message } = (await request.json()) as { message: string };
      // The Stop left before the follow-up, so the server takes it first.
      await waitUntil(() => backend.stopHeld);
      backend.refusedFollowUps.push(message);
      return HttpResponse.json(
        { detail: "Session has no active turn." },
        { status: 409 },
      );
    }),
    http.get(`${PROXIED_SESSION_URL}/messages/pending`, () =>
      HttpResponse.json({ count: 0, messages: [] }),
    ),
    http.post(`${PROXIED_SESSION_URL}/cancel`, async () => {
      backend.stopHeld = true;
      backend.cancels += 1;
      await delay(300);
      return HttpResponse.json({
        cancelled: true,
        reason: null,
        discarded_messages: [],
      });
    }),
  );
  return backend;
}

async function refetchSession() {
  return { data: undefined };
}

async function waitUntil(condition: () => boolean) {
  for (let waited = 0; !condition(); waited += 10) {
    if (waited > 5000) throw new Error("the cancel never reached the server");
    await delay(10);
  }
}

function lastText(messages: UIMessage[]) {
  const last = messages[messages.length - 1];
  if (last?.role !== "assistant") return null;
  return last.parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("")
    .trim();
}

// Held open after the first delta, so only the Stop ends it.
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

function answerChunks(): UIMessageChunk[] {
  return [
    { type: "start", messageId: "next-turn" },
    { type: "start-step" },
    { type: "text-start", id: "next-text" },
    { type: "text-delta", id: "next-text", delta: "Switched to Gmail. " },
    { type: "text-end", id: "next-text" },
    { type: "finish-step" },
    { type: "finish" },
  ];
}
