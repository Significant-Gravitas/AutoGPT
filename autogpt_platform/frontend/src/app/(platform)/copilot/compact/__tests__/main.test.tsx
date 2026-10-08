import {
  getGetV2GetCopilotUsageMockHandler200,
  getGetV2GetPendingMessagesMockHandler200,
  getGetV2GetSessionMockHandler200,
  getGetV2ListSessionsMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getGetV2GetPendingReviewsForChatSessionMockHandler200 } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { server } from "@/mocks/mock-server";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { copilotStreamHandler } from "@/tests/integrations/copilot-sse";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { type ReactNode, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetCopilotChatRegistry } from "../../copilotChatRegistry";
import CompactCopilotPage from "../page";

const BACKEND_URL = "http://localhost:18006";
const SESSION_ID = "compact-session-1";

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => BACKEND_URL,
    },
  };
});

vi.mock("../../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../../helpers")>();
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

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

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
    created_at: `2026-10-08T00:0${sequence}:00Z`,
    metadata: null,
  };
}

function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { retry: false } },
      }),
  );
  return (
    <QueryClientProvider client={queryClient}>
      <NuqsTestingAdapter searchParams={`?sessionId=${SESSION_ID}`}>
        <BackendAPIProvider>
          <OnboardingProvider>
            <TooltipProvider>{children}</TooltipProvider>
          </OnboardingProvider>
        </BackendAPIProvider>
      </NuqsTestingAdapter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  resetCopilotChatRegistry();
  server.use(
    getGetV2GetSessionMockHandler200({
      id: SESSION_ID,
      created_at: "2026-10-08T00:00:00Z",
      updated_at: "2026-10-08T00:00:00Z",
      user_id: "test-user",
      chat_status: "idle",
      messages: [
        sessionMessage(1, "user", "What is on my calendar today?"),
        sessionMessage(
          2,
          "assistant",
          "Two meetings: design review and lunch.",
        ),
      ],
      has_more_messages: false,
      oldest_sequence: null,
      active_stream: null,
      metadata: { dry_run: false, builder_graph_id: null },
      expert_id: null,
    }),
    getGetV2ListSessionsMockHandler200({
      sessions: [
        {
          id: SESSION_ID,
          created_at: "2026-10-08T00:00:00Z",
          updated_at: "2026-10-08T00:00:00Z",
          title: "Calendar check",
          is_processing: false,
        },
      ],
      total: 1,
    }),
    getGetV2GetPendingReviewsForChatSessionMockHandler200([]),
    getGetV2GetPendingMessagesMockHandler200({ count: 0, messages: [] }),
    getGetV2GetCopilotUsageMockHandler200({
      daily: { percent_used: 0, resets_at: new Date("2026-10-09T00:00:00Z") },
      weekly: { percent_used: 0, resets_at: new Date("2026-10-15T00:00:00Z") },
      tier: "PRO",
      reset_cost: 0,
    }),
  );
});

afterEach(() => {
  resetCopilotChatRegistry();
});

describe("CompactCopilotPage", () => {
  it("renders the session, sends a message and shows the streamed reply with tools collapsed", async () => {
    server.use(
      copilotStreamHandler({
        baseUrl: BACKEND_URL,
        sessionId: SESSION_ID,
        chunks: [
          { type: "start", messageId: "reply-1" },
          { type: "start-step" },
          {
            type: "tool-input-start",
            toolCallId: "call-1",
            toolName: "web_search",
          },
          {
            type: "tool-input-available",
            toolCallId: "call-1",
            toolName: "web_search",
            input: { query: "lunch spots near the office" },
          },
          {
            type: "tool-output-available",
            toolCallId: "call-1",
            output: { results: [] },
          },
          { type: "text-start", id: "t1" },
          { type: "text-delta", id: "t1", delta: "Booked a table at Rosa's." },
          { type: "text-end", id: "t1" },
          { type: "finish-step" },
          { type: "finish" },
        ],
      }),
    );

    render(<CompactCopilotPage />, { wrapper: Providers });

    expect(
      await screen.findByText("What is on my calendar today?"),
    ).toBeDefined();
    expect(
      await screen.findByText("Two meetings: design review and lunch."),
    ).toBeDefined();

    const input = screen.getByLabelText(/message otto/i);
    await waitFor(() =>
      expect((input as HTMLTextAreaElement).disabled).toBe(false),
    );

    const user = userEvent.setup();
    await user.type(input, "Book lunch for one");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText("Book lunch for one")).toBeDefined();
    expect(
      await screen.findByText("Booked a table at Rosa's.", undefined, {
        timeout: 5000,
      }),
    ).toBeDefined();
    expect(
      screen.getByText(/searched the web for "lunch spots near the office"/i),
    ).toBeDefined();

    await waitFor(() =>
      expect(screen.getByTestId("agent-activity").textContent).toBe("Done"),
    );
    expect(
      screen.getByRole("img", { name: /otto: done/i }).dataset.status,
    ).toBe("done");
  });
});
