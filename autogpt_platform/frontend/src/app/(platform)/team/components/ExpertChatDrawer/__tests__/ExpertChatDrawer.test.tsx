import {
  getGetV2GetSessionMockHandler200,
  getGetV2ListSessionsMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { ExpertChatDrawer } from "../ExpertChatDrawer";

const EXPERT_ID = "expert-zara";
const SESSION_ID = "session-zara";

describe("ExpertChatDrawer", () => {
  test("a hire's onboarding card is a live form, not a settled row", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          {
            id: SESSION_ID,
            created_at: "2026-09-27T18:00:00Z",
            updated_at: "2026-09-27T18:01:00Z",
            is_processing: false,
            expert_id: EXPERT_ID,
          },
        ],
        total: 1,
      }),
      getGetV2GetSessionMockHandler200({
        id: SESSION_ID,
        created_at: "2026-09-27T18:00:00Z",
        updated_at: "2026-09-27T18:01:00Z",
        user_id: "user-1",
        expert_id: EXPERT_ID,
        messages: onboardingTurn(),
      }),
    );

    render(
      <ExpertChatDrawer
        target={{
          expertId: EXPERT_ID,
          name: "Zara",
          role: "GTM Strategist",
          avatarUrl: null,
        }}
        onClose={() => {}}
      />,
    );

    expect(
      await screen.findByText("Which outcome should I start with?"),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Skip" })).toBeDefined();
    expect(screen.queryByText(/Setup questions from/)).toBeNull();
  });
});

function onboardingTurn(): SessionDetailResponseMessagesItem[] {
  const row = {
    tool_call_id: null,
    tool_calls: null,
    duration_ms: null,
    metadata: null,
  };
  return [
    {
      ...row,
      id: "db-1",
      role: "user",
      content: "Hey!",
      sequence: 1,
      created_at: "2026-09-27T18:00:00Z",
    },
    {
      ...row,
      id: "db-2",
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id: "call-onboarding",
          type: "function",
          function: { name: "expert_onboarding", arguments: "{}" },
        },
      ],
      sequence: 2,
      created_at: "2026-09-27T18:00:10Z",
    },
    {
      ...row,
      id: "db-3",
      role: "tool",
      tool_call_id: "call-onboarding",
      content: JSON.stringify({
        type: "expert_onboarding",
        message: "Which outcome should I start with?",
        session_id: SESSION_ID,
        expert_id: EXPERT_ID,
        greeting: "Hi, I'm Zara.",
        steps: [
          {
            question: "Which outcome should I start with?",
            keyword: "outcome",
            options: ["Positioning", "Pricing"],
          },
        ],
      }),
      sequence: 3,
      created_at: "2026-09-27T18:00:11Z",
    },
    {
      ...row,
      id: "db-4",
      role: "assistant",
      content: "Onboarding card's up — pick your answers above.",
      sequence: 4,
      created_at: "2026-09-27T18:00:12Z",
    },
  ];
}
