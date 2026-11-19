import {
  getGetV2GetAnsweredReviewsWaitingToRunForChatSessionMockHandler,
  getGetV2GetPendingReviewsForChatSessionMockHandler,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { toast } from "@/components/molecules/Toast/use-toast";
import { server } from "@/mocks/mock-server";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useCopilotUIStore } from "../store";
import { getTurnRuntime } from "../stream/turnRuntime";
import {
  renderHost,
  resetChatRuntimes,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

// The turn runtime, the only path a send can go unconfirmed on.
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
    user: { id: "test-user" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
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
vi.mock("@/components/molecules/Toast/use-toast", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: vi.fn(actual.toast) };
});

const STREAM = `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`;
const PENDING = `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/messages/pending`;

function resetStores() {
  resetChatRuntimes();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
  window.localStorage.clear();
}

beforeEach(() => {
  resetStores();
  vi.mocked(toast).mockClear();
});
afterEach(() => {
  resetStores();
  vi.useRealTimers();
});

// Every send stays out, never answered, until the chat is torn down, which
// aborts it. The first is checked and reported.
function holdSends() {
  const sent: { message: string; message_id: string }[] = [];
  const queued: string[] = [];
  server.use(
    getGetV2GetPendingReviewsForChatSessionMockHandler([]),
    getGetV2GetAnsweredReviewsWaitingToRunForChatSessionMockHandler([]),
    http.post(STREAM, async ({ request }) => {
      sent.push((await request.json()) as (typeof sent)[number]);
      await new Promise<void>((resolve) =>
        request.signal.addEventListener("abort", () => resolve(), {
          once: true,
        }),
      );
      return new HttpResponse(null, { status: 204 });
    }),
    http.post(PENDING, async ({ request }) => {
      queued.push(((await request.json()) as { message: string }).message);
      return HttpResponse.json({
        buffer_length: 1,
        max_buffer_length: 10,
        turn_in_flight: true,
      });
    }),
  );
  return { sent, queued };
}

async function reportUnconfirmedSend() {
  // Fake before the chat mounts, so its liveness tick is on this clock.
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const requests = holdSends();
  renderHost();
  await typeAndSend("What does example.com say?");
  // Thirty seconds of silence, then the checks that find no row for it.
  await act(() => vi.advanceTimersByTimeAsync(40_000));
  const sendAgain = await screen.findByRole("button", { name: "Send again" });
  return { ...requests, sendAgain };
}

it("queues a follow-up typed while Send again's request is out, rather than starting a second turn", async () => {
  const { sent, queued, sendAgain } = await reportUnconfirmedSend();

  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  await user.click(sendAgain);
  await waitFor(() => expect(sent).toHaveLength(2));
  // While that request is out the composer queues: what a user does then.
  await user.type(
    screen.getByLabelText(/chat message input/i),
    "And example.org?",
  );
  await user.click(
    await screen.findByRole("button", { name: "Queue message" }),
  );

  await waitFor(() => expect(queued).toEqual(["And example.org?"]));
  expect(sent).toHaveLength(2);
  expect(sent[1].message_id).toBe(sent[0].message_id);
});

it("says so when Send again finds nothing left to send", async () => {
  const { sent, sendAgain } = await reportUnconfirmedSend();

  // A send that lands between the banner's render and the click leaves the
  // runtime nothing to send again.
  act(() => {
    void getTurnRuntime(TEST_SESSION_ID).send(
      { text: "And example.org?" },
      undefined,
    );
    fireEvent.click(sendAgain);
  });

  expect(toast).toHaveBeenCalledWith(
    expect.objectContaining({ title: "Message not sent again" }),
  );
  await waitFor(() => expect(sent).toHaveLength(2));
  expect(sent[1].message).toBe("And example.org?");
});
