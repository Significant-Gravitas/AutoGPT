import { BuilderChatPanel } from "@/app/(platform)/build/components/BuilderChatPanel/BuilderChatPanel";
import { ExpertChatDrawer } from "@/app/(platform)/team/components/ExpertChatDrawer/ExpertChatDrawer";
import {
  getGetV2ListChatTransportsMockHandler,
  getGetV2ListSessionsMockHandler200,
  getPostV2CreateSessionMockHandler,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetMyMemoryOverviewMockHandler200,
  getListMyMemoryFactsMockHandler200,
} from "@/app/api/__generated__/endpoints/memory/memory.msw";
import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { server } from "@/mocks/mock-server";
import { streamSseResponse } from "@/tests/integrations/copilot-sse";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { render as renderWithOwnProviders } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse, ws } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../copilotChatRegistry";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { TEST_BACKEND_BASE_URL } from "./sse-helpers";

// A message sent from a side chat while the user's other chats hold every
// running slot is queued, and starts when one of them ends, which nothing
// tells this tab. Each side chat must say "Queued" and then show the answer,
// as the main chat does, instead of sitting on an unanswered message.

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

vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
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

import SettingsMemoryPage from "@/app/(platform)/settings/memory/page";

const SESSION_ID = "side-chat-capped";
const ANSWER = "Maya is drafting the launch plan.";
// The follow re-reads a queued session every 10 s.
const FOLLOW_MS = 20_000;

let chat: ReturnType<typeof cappedTurnServer>;

beforeEach(() => {
  resetCopilotChatRegistry();
  useCopilotStreamStore.getState().resetAll();
  window.localStorage.clear();
  useAuthStore.setState({
    user: { id: "user-1", email: "side-chat@example.com", user_metadata: {} },
    isUserLoading: false,
    hasLoadedUser: true,
  });
  chat = cappedTurnServer();
  server.use(
    ws.link("ws://localhost:8001/ws").addEventListener("connection", () => {}),
    ...chat.handlers,
  );
});

afterEach(() => {
  useAuthStore.setState({ user: null, hasLoadedUser: false });
});

describe("a side chat's send queued at the running cap", () => {
  it("the memory chat shows it Queued, then its answer once it starts", async () => {
    server.use(
      getListExpertsMockHandler200([]),
      getListMyMemoryFactsMockHandler200({ expert_id: null, items: [] }),
      getGetMyMemoryOverviewMockHandler200({
        expert_id: null,
        facts: 0,
        entities: 0,
        episodes: 0,
      }),
      getPostV2CreateSessionMockHandler(() => created()),
    );
    window.history.replaceState({}, "", "/settings/memory");
    render(<SettingsMemoryPage />);
    // Opening the chat sends its first prompt: that is the queued send.
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Forget a topic…" }));
    const panel = await screen.findByRole("complementary", {
      name: "Memory chat panel",
    });

    await expectQueuedThenAnswered(panel);
  }, 40_000);

  it("the expert chat drawer shows it Queued, then its answer once it starts", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
      getPostV2CreateSessionMockHandler(() => created()),
    );
    render(
      <ExpertChatDrawer
        target={{
          expertId: null,
          name: "Otto",
          role: "autopilot",
          avatarUrl: null,
        }}
        onClose={() => {}}
        resumeLatest={false}
      />,
    );
    await userEvent
      .setup()
      .type(
        await screen.findByPlaceholderText("Message Otto…"),
        "Status please{Enter}",
      );

    await expectQueuedThenAnswered(document.body);
  }, 40_000);

  it("the builder chat shows it Queued, then its answer once it starts", async () => {
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
      getPostV2CreateSessionMockHandler(() => created()),
    );
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    renderWithOwnProviders(<BuilderChatPanel />, {
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
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Chat with builder" }));
    await userEvent
      .setup()
      .type(
        await screen.findByPlaceholderText(
          "Ask the builder to edit or run this workflow…",
          undefined,
          { timeout: 10_000 },
        ),
        "Status please{Enter}",
      );

    await expectQueuedThenAnswered(
      await screen.findByRole("complementary", { name: "Builder chat panel" }),
    );
  }, 40_000);
});

async function expectQueuedThenAnswered(container: HTMLElement) {
  expect(
    await within(container).findByText("Queued", undefined, {
      timeout: 10_000,
    }),
  ).toBeDefined();
  expect(within(container).queryByText(ANSWER)).toBeNull();

  // Another of the user's turns ends and the queued one runs and answers.
  chat.promote();

  expect(
    await within(container).findByText(ANSWER, undefined, {
      timeout: FOLLOW_MS,
    }),
  ).toBeDefined();
  expect(within(container).queryByText("Queued")).toBeNull();
}

function created() {
  return {
    id: SESSION_ID,
    created_at: "2026-10-25T00:00:00Z",
    user_id: "user-1",
  };
}

/** The server side of a send the running cap queued: the POST says so and
 *  stores the row, the session reads "queued" until `promote`, then idle
 *  with the queued turn's answer stored. */
function cappedTurnServer() {
  const state = {
    queued: false,
    stored: [] as SessionDetailResponseMessagesItem[],
  };

  function store(role: "user" | "assistant", content: string) {
    const sequence = state.stored.length + 1;
    state.stored = [
      ...state.stored,
      {
        id: `db-${sequence}`,
        role,
        content,
        tool_call_id: null,
        tool_calls: null,
        sequence,
        duration_ms: null,
        created_at: `2026-10-25T00:${String(sequence).padStart(2, "0")}:00Z`,
        metadata: null,
      },
    ];
  }

  const handlers = [
    http.get(`*/api/chat/sessions/${SESSION_ID}`, () =>
      HttpResponse.json({
        id: SESSION_ID,
        created_at: "2026-10-25T00:00:00Z",
        updated_at: "2026-10-25T00:00:00Z",
        user_id: "user-1",
        chat_status: state.queued ? "queued" : "idle",
        messages: state.stored,
        has_more_messages: false,
        oldest_sequence: null,
        active_stream: null,
        metadata: { dry_run: false, builder_graph_id: null },
        expert_id: null,
      }),
    ),
    http.get(`*/api/chat/sessions/${SESSION_ID}/messages/pending`, () =>
      HttpResponse.json({
        messages: [],
        count: 0,
        items: [],
        held_reason: null,
      }),
    ),
    http.post(
      `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${SESSION_ID}/stream`,
      async ({ request }) => {
        const body = (await request.json()) as { message?: string };
        store("user", body.message ?? "");
        state.queued = true;
        return streamSseResponse(
          [
            { type: "start", messageId: "queued-at-cap" },
            { type: "start-step" },
            { type: "data-turn-queued", data: { position: 1 } },
            { type: "finish-step" },
            { type: "finish" },
          ],
          { abortSignal: request.signal },
        );
      },
    ),
  ];

  return {
    handlers,
    promote() {
      store("assistant", ANSWER);
      state.queued = false;
    },
  };
}
