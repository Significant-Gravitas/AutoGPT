import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
const EXPERT_ID = "expert-zara";
const SESSION_ID = "session-zara";

export function onboardingTurn(): SessionDetailResponseMessagesItem[] {
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
      content: JSON.stringify(onboardingCard()),
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

export function onboardingCard() {
  return {
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
  };
}
