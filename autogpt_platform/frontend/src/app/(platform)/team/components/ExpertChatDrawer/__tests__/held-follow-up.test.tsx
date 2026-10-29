import {
  HELD_FOLLOW_UP,
  heldFollowUpServer,
  RETRY_ANSWER,
} from "@/app/(platform)/copilot/__tests__/heldFollowUpServer";
import { TEST_BACKEND_BASE_URL } from "@/app/(platform)/copilot/__tests__/sse-helpers";
import { useCopilotStreamStore } from "@/app/(platform)/copilot/copilotStreamStore";
import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { ws } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ExpertChatDrawer } from "../ExpertChatDrawer";

// A follow-up sent to an expert while it was busy, left unread when that turn
// ended: the drawer says so and offers Retry, as the main chat does, instead
// of showing it Queued until it expires (SECRT-3156).

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));

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

vi.mock("@/app/(platform)/copilot/helpers", async (importActual) => {
  const actual =
    await importActual<typeof import("@/app/(platform)/copilot/helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

const USER_ID = "user-1";
const EXPERT_ID = "3f8b0f7e-9f30-4a3b-a6a1-000000000001";
const SESSION_ID = "session-zara-held";
const ZARA = {
  expertId: EXPERT_ID,
  name: "Zara",
  role: "GTM Strategist",
  avatarUrl: null,
};

const backendSocket = ws.link("ws://localhost:8001/ws");

let chat: ReturnType<typeof heldFollowUpServer>;

beforeEach(() => {
  window.localStorage.clear();
  useCopilotStreamStore.getState().resetAll();
  useAuthStore.setState({
    user: { id: USER_ID, email: "zara-owner@example.com", user_metadata: {} },
    isUserLoading: false,
    hasLoadedUser: true,
  });
  chat = heldFollowUpServer({
    baseUrl: TEST_BACKEND_BASE_URL,
    sessionId: SESSION_ID,
    heldFrom: "load",
    session: {
      expert_id: EXPERT_ID,
      messages: [
        {
          id: "db-1",
          role: "user",
          content: "Plan the launch",
          sequence: 1,
          created_at: "2026-10-25T00:01:00Z",
        },
        {
          id: "db-2",
          role: "assistant",
          content: "Here is the launch plan.",
          sequence: 2,
          created_at: "2026-10-25T00:02:00Z",
        },
      ],
    },
  });
  server.use(
    backendSocket.addEventListener("connection", () => {}),
    getGetV2ListSessionsMockHandler200({
      sessions: [
        {
          id: SESSION_ID,
          created_at: "2026-10-25T00:00:00Z",
          updated_at: "2026-10-25T00:02:00Z",
          is_processing: false,
          expert_id: EXPERT_ID,
        },
      ],
      total: 1,
    }),
    ...chat.handlers,
  );
});

afterEach(() => {
  useAuthStore.setState({ user: null, hasLoadedUser: false });
});

describe("the expert chat drawer, with a follow-up its last turn left unread", () => {
  test("shows it as not sent with a Retry that resends it by its id", async () => {
    render(<ExpertChatDrawer target={ZARA} onClose={() => {}} />);

    expect(
      await screen.findByText("Didn't go through", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    expect(screen.queryByText("Queued")).toBeNull();

    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: /retry/i }));

    await screen.findByText(RETRY_ANSWER, undefined, { timeout: 5000 });
    expect(chat.resends()).toEqual([
      expect.objectContaining({
        resend_pending_id: "pm-1",
        message_id: "resend-pm-1",
      }),
    ]);
    await waitFor(() =>
      expect(screen.queryByText("Didn't go through")).toBeNull(),
    );
    expect(screen.getAllByText(HELD_FOLLOW_UP)).toHaveLength(1);
  }, 15_000);
});
