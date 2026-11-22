import { getGetV2GetCopilotUsageMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getGetSubscriptionStatusMockHandler200 } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { SESSION_COMPOSER_ID } from "../../SendAtReset/helpers";
import { ChatContainer, type ChatContainerProps } from "../ChatContainer";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

type Message = UIMessage<unknown, UIDataTypes, UITools>;

const EXPERT_ID = "expert-kepler";

const props: ChatContainerProps = {
  messages: [],
  status: "ready",
  error: undefined,
  sessionId: "kickoff-session",
  isLoadingSession: true,
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onStop: vi.fn(),
  onSend: vi.fn(),
  expertIdentity: {
    id: EXPERT_ID,
    name: "Kepler",
    role: "researcher",
    avatarUrl: null,
    isArchived: false,
    readOnlyReason: null,
  },
};

// The composer locks on the AutoPilot budget: pin one with room left instead
// of faker's random usage and tier.
beforeEach(() => {
  sessionStorage.clear();
  server.use(
    getGetV2GetCopilotUsageMockHandler200({
      tier: "PRO",
      daily: null,
      weekly: null,
      reset_cost: 0,
      state: "ok",
      limiting_window: null,
      resets_at: null,
    }),
    getGetSubscriptionStatusMockHandler200({
      tier: "PRO",
      monthly_cost: 0,
      tier_costs: {},
      proration_credit_cents: 0,
    }),
  );
});

describe("keyboard focus when a chat loads", () => {
  test("an unanswered onboarding card keeps focus on its first question", async () => {
    const { rerender } = render(<ChatContainer {...props} />);
    rerender(
      <ChatContainer
        {...props}
        isLoadingSession={false}
        messages={[kickoffMessage(), onboardingMessage()]}
      />,
    );
    const firstOption = await screen.findByRole("radio", { name: /Pricing/ });
    await waitFor(() => expect(document.activeElement).toBe(firstOption));
  });

  test("the composer takes focus when the last message is not a card", async () => {
    const { rerender } = render(<ChatContainer {...props} />);
    rerender(
      <ChatContainer
        {...props}
        isLoadingSession={false}
        messages={[kickoffMessage(), replyMessage()]}
      />,
    );
    expect(await screen.findByText("Ready when you are.")).toBeDefined();
    await waitFor(() =>
      expect(document.activeElement?.id).toBe(SESSION_COMPOSER_ID),
    );
  });
});

function kickoffMessage(): Message {
  return {
    id: "user-1",
    role: "user",
    parts: [{ type: "text", text: "Hey!" }],
  };
}

function onboardingMessage(): Message {
  return {
    id: "assistant-1",
    role: "assistant",
    parts: [
      {
        type: "tool-expert_onboarding",
        toolCallId: "call-onboarding",
        state: "output-available",
        input: {},
        output: {
          type: "expert_onboarding",
          message: "Which outcome should I start with?",
          session_id: "kickoff-session",
          expert_id: EXPERT_ID,
          greeting: "Hi, I'm Kepler.",
          steps: [
            {
              question: "Which outcome should I start with?",
              keyword: "outcome",
              options: ["Pricing", "Positioning"],
            },
          ],
        },
      },
    ],
  } as Message;
}

function replyMessage(): Message {
  return {
    id: "assistant-1",
    role: "assistant",
    parts: [{ type: "text", text: "Ready when you are." }],
  };
}
