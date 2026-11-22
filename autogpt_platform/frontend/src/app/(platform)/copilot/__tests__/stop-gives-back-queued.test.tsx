import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import {
  createStreamHold,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { act, screen, waitFor } from "@testing-library/react";
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

// Stop discards the follow-ups queued before it on the server, and their text
// goes back to the composer of the chat that was stopped.

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
  sessionStorage.clear();
  // Composer drafts are kept per signed-in user and chat.
  useAuthStore.setState({
    user: {
      id: "test-user",
      email: "test-user@example.com",
      user_metadata: {},
    },
    isUserLoading: false,
    hasLoadedUser: true,
  });
});

afterEach(() => {
  resetChatRuntimes();
  server.resetHandlers();
});

describe.each(STREAM_PATHS)("on the %s path", (path) => {
  beforeEach(() => {
    streamPath.runtime = path === "stream runtime";
  });

  // The page re-reads the buffer the moment it stops, while the cancel is
  // still clearing it; either answer must leave the strip empty.
  it.each(["before", "after"])(
    "Stop puts a follow-up queued before it back in the composer (re-read answered %s the cancel)",
    async (order) => {
      const { buffer } = await queueAFollowUpWhileOttoAnswers();
      const cancel = hold();
      buffer.cancelGate = cancel.until;
      const reread = hold();
      if (order === "after") buffer.peekGate = reread.until;
      const askedBeforeStop = buffer.peeksAsked;

      await clickStop();
      await waitFor(() =>
        expect(buffer.peeksAsked).toBeGreaterThan(askedBeforeStop),
      );
      if (order === "before") {
        await waitFor(() =>
          expect(buffer.newestPeekAnswered).toBeGreaterThan(askedBeforeStop),
        );
        cancel.release();
      } else {
        cancel.release();
        await waitFor(() => expect(buffer.cancelsAnswered).toBe(1));
        reread.release();
      }

      await waitFor(() => expect(composer().value).toBe(QUEUED), {
        timeout: 5000,
      });
      await waitFor(
        () =>
          expect(buffer.newestPeekAnswered).toBeGreaterThan(askedBeforeStop),
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
    const { buffer } = await queueAFollowUpWhileOttoAnswers();
    buffer.messages = [];
    const cancels = buffer.cancelsAnswered;
    // Only the Stop may clear the chip here, not a re-read of the empty buffer.
    const reread = hold();
    buffer.peekGate = reread.until;

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
    reread.release();
  }, 20_000);

  it("a Stop that can't tell what it discarded leaves the queued follow-up shown", async () => {
    const { buffer } = await queueAFollowUpWhileOttoAnswers();
    buffer.discardsKnown = false;
    buffer.cancelReason = "cancel_published_not_confirmed";
    const cancel = hold();
    buffer.cancelGate = cancel.until;

    await clickStop();
    // The Stop raises this toast in the same tick as it hands the page what it
    // discarded, and act renders the page's answer to that before it returns.
    await act(async () => {
      cancel.release();
      await vi.waitFor(
        () =>
          expect(toast).toHaveBeenCalledWith(
            expect.objectContaining({ title: "Stop may take a moment" }),
          ),
        { timeout: 5000 },
      );
    });

    expect(screen.getByText("Queued")).toBeDefined();
    expect(composer().value).toBe("");
    expect(toast).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "Queued message not sent" }),
    );
  }, 20_000);

  it("a follow-up still being sent at the Stop loses its chip once the server discards it", async () => {
    const { buffer } = await ottoAnswers();
    const send = hold();
    buffer.sendGate = send.until;
    await sendFollowUp();
    await waitFor(() => expect(buffer.messages).toEqual([QUEUED]));
    const cancel = hold();
    buffer.cancelGate = cancel.until;
    // The re-read at the Stop answers after the cancel, so it can't clear the chip.
    const reread = hold();
    buffer.peekGate = reread.until;

    await clickStop();
    send.release();
    await screen.findByText("Queued");
    cancel.release();

    await waitFor(() => expect(composer().value).toBe(QUEUED), {
      timeout: 5000,
    });
    expect(screen.queryByText("Queued")).toBeNull();
    reread.release();
  }, 20_000);

  it("a follow-up still being sent at the Stop leaves no chip when the re-read finds it first", async () => {
    // The re-read at the Stop answers before both the send and the cancel.
    const { buffer } = await ottoAnswers();
    const send = hold();
    buffer.sendGate = send.until;
    await sendFollowUp();
    await waitFor(() => expect(buffer.messages).toEqual([QUEUED]));
    const cancel = hold();
    buffer.cancelGate = cancel.until;

    const askedBeforeStop = buffer.peeksAsked;
    await clickStop();
    await waitFor(
      () => expect(buffer.newestPeekAnswered).toBeGreaterThan(askedBeforeStop),
      { timeout: 5000 },
    );
    send.release();
    await screen.findByText("Queued");
    cancel.release();

    await waitFor(() => expect(composer().value).toBe(QUEUED), {
      timeout: 5000,
    });
    expect(buffer.cancelsAnswered).toBe(1);
    expect(screen.queryAllByText("Queued")).toEqual([]);
  }, 20_000);

  it("a follow-up given back after a switch to another chat stays with the stopped chat", async () => {
    const { buffer, host } = await queueAFollowUpWhileOttoAnswers();
    const cancel = hold();
    buffer.cancelGate = cancel.until;

    await clickStop();
    host.unmount();
    const other = renderHost({ searchParams: "?sessionId=another-chat" });
    cancel.release();
    await waitFor(() => expect(buffer.cancelsAnswered).toBe(1), {
      timeout: 5000,
    });
    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Queued message not sent" }),
      ),
    );
    expect(composer().value).toBe("");

    other.unmount();
    renderHost();
    await waitFor(() => expect(composer().value).toBe(QUEUED), {
      timeout: 5000,
    });
  }, 20_000);
});

async function queueAFollowUpWhileOttoAnswers() {
  const { buffer, host } = await ottoAnswers();
  await sendFollowUp();
  await screen.findByText("Queued");
  expect(buffer.messages).toEqual([QUEUED]);
  return { buffer, host };
}

async function ottoAnswers() {
  const { hold } = createStreamHold(4);
  server.use(
    http.post(`${SESSION_URL}/stream`, ({ request }) =>
      streamSseResponse(stoppableTurnChunks(), {
        hold,
        abortSignal: request.signal,
      }),
    ),
  );
  const host = renderHost();
  const buffer = serverPendingBuffer();

  await typeAndSend(FIRST);
  await screen.findByText(LIVE_PARTIAL, undefined, { timeout: 5000 });
  return { buffer, host };
}

async function sendFollowUp() {
  const user = userEvent.setup();
  await user.type(composer(), QUEUED);
  await user.click(screen.getByRole("button", { name: "Queue message" }));
}

// The pending buffer as the server holds it; the cancel reports what it
// discarded. A test holds a gate to set the order the re-read, the send and
// the cancel answer in.
function serverPendingBuffer() {
  const buffer = {
    messages: [] as string[],
    peeksAsked: 0,
    // Which re-read, counting from the first asked, answered most recently.
    newestPeekAnswered: 0,
    cancelsAnswered: 0,
    // False when the cancel's own step failed and it can't say what it took.
    discardsKnown: true,
    cancelReason: null as string | null,
    // A re-read reads the buffer when asked and answers once this resolves.
    peekGate: Promise.resolve(),
    // The cancel also answers only after the send.
    sendGate: Promise.resolve(),
    cancelGate: Promise.resolve(),
  };
  server.use(
    http.post(`${SESSION_URL}/messages/pending`, async ({ request }) => {
      const { message } = (await request.json()) as { message: string };
      buffer.messages = [...buffer.messages, message];
      await buffer.sendGate;
      return HttpResponse.json({
        buffer_length: buffer.messages.length,
        max_buffer_length: 10,
        turn_in_flight: true,
      });
    }),
    http.get(`${PROXIED_SESSION_URL}/messages/pending`, async () => {
      buffer.peeksAsked += 1;
      const asked = buffer.peeksAsked;
      const messages = buffer.messages;
      await buffer.peekGate;
      buffer.newestPeekAnswered = Math.max(buffer.newestPeekAnswered, asked);
      return HttpResponse.json({ count: messages.length, messages });
    }),
    http.post(`${PROXIED_SESSION_URL}/cancel`, async () => {
      await buffer.sendGate;
      await buffer.cancelGate;
      const discarded = buffer.messages;
      buffer.messages = [];
      buffer.cancelsAnswered += 1;
      return HttpResponse.json({
        cancelled: true,
        reason: buffer.cancelReason,
        discarded_messages: buffer.discardsKnown ? discarded : null,
      });
    }),
  );
  return buffer;
}

function hold() {
  let release = () => {};
  const until = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { until, release };
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
