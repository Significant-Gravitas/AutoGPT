import type { TourScript } from "./types";

export const monitorPricingScript: TourScript = [
  {
    assistantMessageId: "tour-asst-1",
    steps: [
      {
        delayMs: 400,
        part: {
          type: "text",
          text: "Great goal! Let me break that down into steps.",
          state: "done",
        },
      },
      {
        delayMs: 700,
        part: {
          type: "tool-decompose_goal",
          toolCallId: "tour-call-decompose",
          state: "output-available",
          input: {
            goal: "Watch a competitor's pricing page and email me on change",
          },
          output: {
            steps: [
              "Fetch the competitor pricing page on a schedule",
              "Detect changes vs. the last snapshot",
              "Email me a summary when it changes",
            ],
          },
        },
      },
      {
        delayMs: 500,
        part: {
          type: "tool-TodoWrite",
          toolCallId: "tour-call-todo",
          state: "output-available",
          input: {
            todos: [
              {
                content: "Fetch pricing page",
                status: "completed",
                activeForm: "Fetching pricing page",
              },
              {
                content: "Detect price changes",
                status: "in_progress",
                activeForm: "Detecting price changes",
              },
              {
                content: "Send email alert",
                status: "pending",
                activeForm: "Sending email alert",
              },
            ],
          },
          output: { ok: true },
        },
      },
    ],
  },
  {
    assistantMessageId: "tour-asst-2",
    steps: [
      {
        delayMs: 500,
        part: {
          type: "text",
          text: "I'll build that agent for you now.",
          state: "done",
        },
      },
      {
        delayMs: 900,
        part: {
          type: "tool-create_agent",
          toolCallId: "tour-call-create",
          state: "output-available",
          input: { name: "Competitor Pricing Watcher" },
          output: {
            agent_id: "tour-agent-1",
            name: "Competitor Pricing Watcher",
            graph: { nodes: 4, edges: 3 },
          },
        },
      },
      {
        delayMs: 700,
        part: {
          type: "tool-run_agent",
          toolCallId: "tour-call-run",
          state: "output-available",
          input: { agent_id: "tour-agent-1" },
          output: {
            status: "completed",
            summary: "No price change detected yet. You're all set!",
          },
        },
      },
      {
        delayMs: 500,
        part: {
          type: "text",
          text: "Done — your agent is live and will email you on any change. 🎉",
          state: "done",
        },
      },
    ],
  },
];
