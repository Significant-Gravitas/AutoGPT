import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import {
  HELD_FOLLOW_UP,
  heldFollowUpServer,
  RETRY_ANSWER,
} from "@/app/(platform)/copilot/__tests__/heldFollowUpServer";
import { TEST_BACKEND_BASE_URL } from "@/app/(platform)/copilot/__tests__/sse-helpers";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getPostV2CreateSessionMockHandler200,
  getPostV2CreateSessionResponseMock200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getGetMyMemoryOverviewMockHandler200,
  getListMyMemoryFactsMockHandler200,
} from "@/app/api/__generated__/endpoints/memory/memory.msw";
import { server } from "@/mocks/mock-server";

// A follow-up typed in the memory chat while Otto was busy, left unread when
// that turn ended: the panel says so and offers Retry, as the main chat does,
// instead of showing it Queued until it expires (SECRT-3156).

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

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

import SettingsMemoryPage from "../page";

const SESSION_ID = "memory-held-session";

let chat: ReturnType<typeof heldFollowUpServer>;

beforeEach(() => {
  resetCopilotChatRegistry();
  window.history.replaceState({}, "", "/settings/memory");
  chat = heldFollowUpServer({
    baseUrl: TEST_BACKEND_BASE_URL,
    sessionId: SESSION_ID,
    heldFrom: "first turn",
  });
  server.use(
    getListExpertsMockHandler200([]),
    getListMyMemoryFactsMockHandler200({ expert_id: null, items: [] }),
    getGetMyMemoryOverviewMockHandler200({
      expert_id: null,
      facts: 0,
      entities: 0,
      episodes: 0,
    }),
    getPostV2CreateSessionMockHandler200(
      getPostV2CreateSessionResponseMock200({ id: SESSION_ID }),
    ),
    ...chat.handlers,
  );
});

describe("the memory chat, with a follow-up its last turn left unread", () => {
  it("shows it as not sent with a Retry that resends it by its id", async () => {
    render(<SettingsMemoryPage />);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Forget a topic…" }));
    const panel = await screen.findByRole("complementary", {
      name: "Memory chat panel",
    });

    expect(
      await within(panel).findByText("Didn't go through", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    expect(within(panel).queryByText("Queued")).toBeNull();

    await userEvent
      .setup()
      .click(within(panel).getByRole("button", { name: /retry/i }));

    await within(panel).findByText(RETRY_ANSWER, undefined, { timeout: 5000 });
    expect(chat.resends()).toEqual([
      expect.objectContaining({
        resend_pending_id: "pm-1",
        message_id: "resend-pm-1",
      }),
    ]);
    await waitFor(() =>
      expect(within(panel).queryByText("Didn't go through")).toBeNull(),
    );
    expect(within(panel).getAllByText(HELD_FOLLOW_UP)).toHaveLength(1);
  }, 15_000);
});
