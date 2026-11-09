import { server } from "@/mocks/mock-server";
import {
  createStreamHold,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UIMessageChunk } from "ai";
import { delay, http, HttpResponse } from "msw";
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

// Stop discards the follow-ups queued before it, on the server. The chat kept
// drawing one as "Queued" until a reload, or dropped it without a word
// (SECRT-3127); its text goes back to the composer instead.

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
      flag === actual.Flag.COPILOT_STREAM_RUNTIME ? streamPath.runtime : false,
  };
});

const toast = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast };
});

const FIRST = "Plan my week";
const QUEUED = "Also check Friday";
const LIVE_PARTIAL = "Looking at your calendar";
const SESSION_URL = `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}`;
// The generated client reaches these through the app's proxy.
const PROXIED_SESSION_URL = `*/api/chat/sessions/${TEST_SESSION_ID}`;

beforeEach(() => {
  resetChatRuntimes();
  toast.mockClear();
});

afterEach(() => {
  resetChatRuntimes();
  server.resetHandlers();
});

describe.each(STREAM_PATHS)("on the %s path", (path) => {
  beforeEach(() => {
    streamPath.runtime = path === "stream runtime";
  });

  // The page re-reads the buffer the moment it stops, racing the cancel's
  // clear; either answer must leave the strip empty.
  it.each([
    ["before", 0],
    ["after", 600],
  ])(
    "Stop puts a follow-up queued before it back in the composer (re-read answered %s the cancel)",
    async (_order, peekDelayMs) => {
      const buffer = await queueAFollowUpWhileOttoAnswers(peekDelayMs);

      const peeksBeforeStop = buffer.peeksAnswered;
      await clickStop();

      await waitFor(() => expect(composer().value).toBe(QUEUED), {
        timeout: 5000,
      });
      await waitFor(
        () => expect(buffer.peeksAnswered).toBeGreaterThan(peeksBeforeStop),
        { timeout: 5000 },
      );
      expect(buffer.messages).toEqual([]);
      expect(screen.queryByText("Queued")).toBeNull();
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Queued message not sent" }),
      );
    },
    20_000,
  );

  it("Stop gives nothing back when the stopped turn already read the follow-up", async () => {
    const buffer = await queueAFollowUpWhileOttoAnswers(0);
    buffer.messages = [];
    const cancels = buffer.cancelsAnswered;

    await clickStop();

    await waitFor(() => expect(buffer.cancelsAnswered).toBe(cancels + 1), {
      timeout: 5000,
    });
    await waitFor(() => expect(screen.queryByText("Queued")).toBeNull(), {
      timeout: 5000,
    });
    expect(composer().value).toBe("");
    expect(toast).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "Queued message not sent" }),
    );
  }, 20_000);
});

async function queueAFollowUpWhileOttoAnswers(peekDelayMs: number) {
  const { hold } = createStreamHold(4);
  server.use(
    http.post(`${SESSION_URL}/stream`, ({ request }) =>
      streamSseResponse(stoppableTurnChunks(), {
        hold,
        abortSignal: request.signal,
      }),
    ),
  );
  renderHost();
  const buffer = serverPendingBuffer(peekDelayMs);

  await typeAndSend(FIRST);
  await screen.findByText(LIVE_PARTIAL, undefined, { timeout: 5000 });
  const user = userEvent.setup();
  await user.type(composer(), QUEUED);
  await user.click(screen.getByRole("button", { name: "Queue message" }));
  await screen.findByText("Queued");
  expect(buffer.messages).toEqual([QUEUED]);
  return buffer;
}

// The pending buffer as the server holds it. The cancel answers a moment late,
// as it does while the executor tears the turn down, so a re-read made at the
// Stop still sees the follow-up there; it reports what it discarded.
function serverPendingBuffer(peekDelayMs: number) {
  const buffer = {
    messages: [] as string[],
    peeksAnswered: 0,
    cancelsAnswered: 0,
  };
  server.use(
    http.post(`${SESSION_URL}/messages/pending`, async ({ request }) => {
      const { message } = (await request.json()) as { message: string };
      buffer.messages = [...buffer.messages, message];
      return HttpResponse.json({
        buffer_length: buffer.messages.length,
        max_buffer_length: 10,
        turn_in_flight: true,
      });
    }),
    http.get(`${PROXIED_SESSION_URL}/messages/pending`, async () => {
      const messages = buffer.messages;
      await delay(peekDelayMs);
      buffer.peeksAnswered += 1;
      return HttpResponse.json({ count: messages.length, messages });
    }),
    http.post(`${PROXIED_SESSION_URL}/cancel`, async () => {
      await delay(300);
      const discarded = buffer.messages;
      buffer.messages = [];
      buffer.cancelsAnswered += 1;
      return HttpResponse.json({
        cancelled: true,
        reason: null,
        discarded_messages: discarded,
      });
    }),
  );
  return buffer;
}

function composer() {
  return screen.getByLabelText(/chat message input/i) as HTMLTextAreaElement;
}

// Held open after the first delta, so only the Stop click ends it.
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
