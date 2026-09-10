import { RUN_DELAY_MS } from "./constants";
import type { TourScript } from "./types";

export const dailyBriefScript: TourScript = [
  {
    assistantMessageId: "brief-asst-1",
    userPrompt:
      "Every morning, pull my unread emails and calendar into one brief",
    steps: [
      {
        delayMs: 400,
        part: {
          type: "text",
          text: "Love it. Here's how I'll set that up.",
        },
      },
      {
        delayMs: 700,
        part: {
          type: "plan",
          plan: {
            goal: "Send me one morning brief with my email and calendar",
            steps: [
              {
                description: "Read unread emails and today's calendar",
                blockName: "Gmail Read",
              },
              {
                description: "Summarize everything into a short brief",
                blockName: "AI Text Generator",
              },
              {
                description: "Deliver it every morning at 7:30",
                blockName: "Send Email",
              },
            ],
          },
        },
      },
      {
        delayMs: 500,
        part: {
          type: "text",
          text: "Want me to build and run it right now?",
        },
      },
    ],
  },
  {
    assistantMessageId: "brief-asst-2",
    userPrompt: "Perfect, set it up",
    steps: [
      {
        delayMs: 500,
        part: {
          type: "text",
          text: "Building your agent — picking blocks, wiring them up, setting the schedule.",
        },
      },
      {
        delayMs: 900,
        part: {
          type: "agent",
          agent: {
            name: "Morning Brief",
            schedule: "Daily · 7:30 AM",
            blocks: ["Gmail Read", "AI Text Generator", "Send Email"],
          },
        },
      },
      {
        delayMs: RUN_DELAY_MS,
        part: {
          type: "text",
          text: "Done — first run just finished. Here's what it produced:",
        },
      },
      {
        delayMs: 700,
        part: {
          type: "artifact",
          artifact: {
            caption: "what lands in your inbox",
            title: "Your brief — Thu, Jul 2",
            bullets: [
              "9:30 AM roadmap review — deck still unshared",
              "3 unread from Acme Corp, contract redlines attached",
              "Reply owed: Jordan on the Q3 pricing proposal",
            ],
          },
        },
      },
      {
        delayMs: 600,
        part: {
          type: "text",
          text: "All set — a brief like this will be waiting in your inbox at 7:30 every morning. 🎉",
        },
      },
    ],
  },
];
