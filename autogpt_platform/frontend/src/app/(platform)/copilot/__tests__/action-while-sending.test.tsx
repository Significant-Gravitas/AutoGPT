import {
  getGetV2GetAnsweredReviewsWaitingToRunForChatSessionMockHandler,
  getGetV2GetPendingReviewsForChatSessionMockHandler,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { toast } from "@/components/molecules/Toast/use-toast";
import { server } from "@/mocks/mock-server";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useCopilotUIStore } from "../store";
import {
  renderHost,
  resetChatRuntimes,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  waitForInputReady,
} from "./sse-helpers";

// The turn runtime, the path whose sends refuse a second request.
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
const TITLE = "the Printify order pipeline";
const PROMPT = "Draft this week's report";

// A hire whose hand-off didn't start, and that the backend says a retry can
// help: its card offers Try again.
const messages: SessionDetailResponseMessagesItem[] = [
  { role: "user", content: "Hire someone for my Printify orders", sequence: 0 },
  {
    role: "assistant",
    content: "",
    sequence: 1,
    tool_calls: [
      {
        id: "call-confirm",
        type: "function",
        function: { name: "confirm_expert_change", arguments: "{}" },
      },
    ],
  },
  {
    role: "tool",
    tool_call_id: "call-confirm",
    sequence: 2,
    content: JSON.stringify({
      type: "expert_change_applied",
      applied: true,
      kind: "hire",
      expert: { id: "exp-james", name: "James", role: "Operations" },
      handoff: { title: TITLE, status: "failed", link: null, retry: true },
    }),
  },
  { role: "assistant", content: "James is on your team.", sequence: 3 },
];

function resetStores() {
  resetChatRuntimes();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
  window.localStorage.clear();
}

beforeEach(resetStores);
afterEach(resetStores);

it("queues a hand-off's Try again clicked while a send is still out, and leaves nothing unhandled", async () => {
  const sent: string[] = [];
  const queued: string[] = [];
  server.use(
    getGetV2GetPendingReviewsForChatSessionMockHandler([]),
    getGetV2GetAnsweredReviewsWaitingToRunForChatSessionMockHandler([]),
    // James has no chat yet, so the card offers Try again, not a link.
    http.get("*/api/chat/sessions", () =>
      HttpResponse.json({ sessions: [], total: 0 }),
    ),
    http.post(STREAM, async ({ request }) => {
      sent.push(((await request.json()) as { message: string }).message);
      // Never answered: the send stays out until the chat is torn down.
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
  const unhandled: unknown[] = [];
  const record = (reason: unknown) => unhandled.push(reason);
  process.on("unhandledRejection", record);
  try {
    renderHost({ sessionOverride: { messages } });
    await screen.findByRole("button", { name: "Try again" });
    await waitForInputReady();
    await userEvent
      .setup()
      .type(screen.getByLabelText(/chat message input/i), PROMPT);

    // Back to back, with no timer able to run between them: the second click
    // lands while the first send's request is open.
    fireEvent.click(screen.getByRole("button", { name: /submit/i }));
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    await waitFor(() =>
      expect(queued).toEqual([`Try handing ${TITLE} to James again.`]),
    );
    expect(sent).toEqual([PROMPT]);
    expect(screen.getByText("Asked Otto to try again.")).toBeDefined();
    expect(toast).not.toHaveBeenCalled();
    expect(unhandled).toEqual([]);
  } finally {
    process.off("unhandledRejection", record);
  }
});
