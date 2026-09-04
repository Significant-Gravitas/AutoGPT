import { RUN_DELAY_MS } from "./constants";
import type { TourScript } from "./types";

export const competitorWatchScript: TourScript = [
  {
    assistantMessageId: "competitor-asst-1",
    userPrompt:
      "Watch a competitor's pricing page and email me when the price changes",
    steps: [
      {
        delayMs: 400,
        part: {
          type: "text",
          text: "Great goal! Let me break that down into steps.",
        },
      },
      {
        delayMs: 700,
        part: {
          type: "plan",
          plan: {
            goal: "Watch a competitor's pricing page and email me on change",
            steps: [
              {
                description: "Fetch the competitor pricing page on a schedule",
                blockName: "Send Web Request",
              },
              {
                description: "Detect changes vs. the last snapshot",
                blockName: "Text Compare",
              },
              {
                description: "Email me a summary when it changes",
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
    assistantMessageId: "competitor-asst-2",
    userPrompt: "Yes, build and run it for me",
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
            name: "Competitor Pricing Watcher",
            schedule: "Daily · 8:00 AM",
            blocks: ["Send Web Request", "Text Compare", "Send Email"],
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
            title: "Price change detected — Competitor Pro plan",
            subtitle: "Pro tier changed on competitor.com/pricing at 8:02 AM:",
            diff: { from: "$49/mo", to: "$59/mo", delta: "+20.4%" },
          },
        },
      },
      {
        delayMs: 600,
        part: {
          type: "text",
          text: "Your agent is live — it'll email you the moment that price moves again. 🎉",
        },
      },
    ],
  },
];
