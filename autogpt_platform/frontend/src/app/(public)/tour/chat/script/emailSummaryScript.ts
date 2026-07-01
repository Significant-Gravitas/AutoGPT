import type { TourScript } from "./types";

export const emailSummaryScript: TourScript = [
  {
    assistantMessageId: "email-asst-1",
    userPrompt: "Summarize my unread emails every morning and send me a digest",
    steps: [
      {
        delayMs: 400,
        part: {
          type: "text",
          text: "Love it. Here's how I'll set that up.",
          state: "done",
        },
      },
      {
        delayMs: 700,
        part: {
          type: "tool-decompose_goal",
          toolCallId: "email-call-decompose",
          state: "output-available",
          input: {
            goal: "Summarize my unread emails every morning and send a digest",
          },
          output: {
            type: "task_decomposition",
            message: "Here's the plan (3 steps):",
            goal: "Summarize my unread emails every morning and send a digest",
            step_count: 3,
            steps: [
              {
                step_id: "step_1",
                description: "Read unread emails from the last 24 hours",
                action: "add_block",
                block_name: "Gmail Read",
                status: "completed",
              },
              {
                step_id: "step_2",
                description: "Summarize each thread into a short digest",
                action: "add_block",
                block_name: "AI Text Generator",
                status: "completed",
              },
              {
                step_id: "step_3",
                description: "Email me the digest every morning at 8am",
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
          toolCallId: "email-call-todo",
          state: "output-available",
          input: {
            todos: [
              {
                content: "Read unread emails",
                status: "completed",
                activeForm: "Reading unread emails",
              },
              {
                content: "Summarize threads",
                status: "in_progress",
                activeForm: "Summarizing threads",
              },
              {
                content: "Send morning digest",
                status: "pending",
                activeForm: "Sending morning digest",
              },
            ],
          },
          output: { ok: true },
        },
      },
    ],
  },
  {
    assistantMessageId: "email-asst-2",
    userPrompt: "Perfect, set it up",
    steps: [
      {
        delayMs: 500,
        part: {
          type: "text",
          text: "Building your digest agent now.",
          state: "done",
        },
      },
      {
        delayMs: 900,
        part: {
          type: "tool-create_agent",
          toolCallId: "email-call-create",
          state: "output-available",
          input: { name: "Weekly Email Digest" },
          output: {
            type: "agent_builder_preview",
            message: "Here's the agent I built for you:",
            agent_name: "Morning Email Digest",
            description:
              "Reads your unread emails, summarizes them, and emails you a digest every morning.",
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
          toolCallId: "email-call-run",
          state: "output-available",
          input: { agent_id: "email-agent-1" },
          output: {
            type: "agent_output",
            message: "Your agent ran successfully.",
            agent_name: "Morning Email Digest",
            agent_id: "email-agent-1",
            execution: {
              execution_id: "email-exec-1",
              status: "COMPLETED",
              outputs: {
                summary:
                  "Digest scheduled. You'll get your first summary tomorrow at 8am.",
              },
            },
          },
        },
      },
      {
        delayMs: 500,
        part: {
          type: "text",
          text: "All set! Your morning digest is scheduled and ready to go. 🎉",
          state: "done",
        },
      },
    ],
  },
];
