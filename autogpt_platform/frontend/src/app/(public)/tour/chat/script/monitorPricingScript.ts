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
            type: "task_decomposition",
            message: "Here's the plan (3 steps):",
            goal: "Watch a competitor's pricing page and email me on change",
            step_count: 3,
            steps: [
              {
                step_id: "step_1",
                description: "Fetch the competitor pricing page on a schedule",
                action: "add_block",
                block_name: "Send Web Request",
                status: "completed",
              },
              {
                step_id: "step_2",
                description: "Detect changes vs. the last snapshot",
                action: "add_block",
                block_name: "Text Compare",
                status: "completed",
              },
              {
                step_id: "step_3",
                description: "Email me a summary when it changes",
                action: "add_block",
                block_name: "Send Email",
                status: "completed",
              },
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
            type: "agent_builder_preview",
            message: "Here's the agent I built for you:",
            agent_name: "Competitor Pricing Watcher",
            description:
              "Checks a competitor's pricing page on a schedule and emails you when it changes.",
            node_count: 4,
            link_count: 3,
            agent_json: { nodes: [], links: [] },
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
            type: "agent_output",
            message: "Your agent ran successfully.",
            agent_name: "Competitor Pricing Watcher",
            agent_id: "tour-agent-1",
            execution: {
              execution_id: "tour-exec-1",
              status: "COMPLETED",
              outputs: {
                summary:
                  "No price change detected yet. You're all set, and I'll email you the moment it changes.",
              },
            },
          },
        },
      },
      {
        delayMs: 500,
        part: {
          type: "text",
          text: "Done! Your agent is live and will email you on any change. 🎉",
          state: "done",
        },
      },
    ],
  },
];
