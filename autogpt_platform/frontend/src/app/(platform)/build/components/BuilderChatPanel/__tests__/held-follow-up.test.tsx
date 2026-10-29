import {
  HELD_FOLLOW_UP,
  heldFollowUpServer,
  RETRY_ANSWER,
} from "@/app/(platform)/copilot/__tests__/heldFollowUpServer";
import { TEST_BACKEND_BASE_URL } from "@/app/(platform)/copilot/__tests__/sse-helpers";
import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import {
  getGetV2ListChatTransportsMockHandler,
  getPostV2CreateSessionMockHandler,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { server } from "@/mocks/mock-server";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { BuilderChatPanel } from "../BuilderChatPanel";

// A follow-up typed in the builder chat while it was busy, left unread when
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

const SESSION_ID = "builder-held-session";

let chat: ReturnType<typeof heldFollowUpServer>;

beforeEach(() => {
  resetCopilotChatRegistry();
  chat = heldFollowUpServer({
    baseUrl: TEST_BACKEND_BASE_URL,
    sessionId: SESSION_ID,
    heldFrom: "load",
    session: { metadata: { dry_run: false, builder_graph_id: "graph-1" } },
  });
  server.use(
    getGetV2ListChatTransportsMockHandler({
      transports: [
        {
          auth_provider: "platform",
          credential_id: null,
          label: "Platform chat",
          available: true,
          default: true,
        },
      ],
    }),
    getPostV2CreateSessionMockHandler(() => ({
      id: SESSION_ID,
      created_at: "2026-10-25T00:00:00Z",
      user_id: "user-1",
    })),
    ...chat.handlers,
  );
});

function renderPanel() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<BuilderChatPanel />, {
    wrapper: ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <NuqsTestingAdapter searchParams="?flowID=graph-1">
          <BackendAPIProvider>
            <OnboardingProvider>
              <TooltipProvider>{children}</TooltipProvider>
            </OnboardingProvider>
          </BackendAPIProvider>
        </NuqsTestingAdapter>
      </QueryClientProvider>
    ),
  });
}

describe("the builder chat, with a follow-up its last turn left unread", () => {
  it("shows it as not sent with a Retry that resends it by its id", async () => {
    renderPanel();
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Chat with builder" }));
    const panel = await screen.findByRole("complementary", {
      name: "Builder chat panel",
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
