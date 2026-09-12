import { RUN_DELAY_MS } from "./constants";
import type { TourScript } from "./types";

export const callPrepScript: TourScript = [
  {
    assistantMessageId: "callprep-asst-1",
    userPrompt: "Brief me on the company and person before every sales call",
    steps: [
      {
        delayMs: 400,
        part: {
          type: "text",
          text: "Great use case. Here's how I'll set that up.",
        },
      },
      {
        delayMs: 700,
        part: {
          type: "plan",
          plan: {
            goal: "Prep me before each external call on my calendar",
            steps: [
              {
                description: "Spot upcoming external calls on my calendar",
                blockName: "Google Calendar",
              },
              {
                description: "Research the company and attendees",
                blockName: "Web Search",
              },
              {
                description: "Send a one-page brief 30 minutes before",
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
    assistantMessageId: "callprep-asst-2",
    userPrompt: "Sounds great, set it up",
    steps: [
      {
        delayMs: 500,
        part: {
          type: "text",
          text: "Building your agent — picking blocks, wiring them up, setting the trigger.",
        },
      },
      {
        delayMs: 900,
        part: {
          type: "agent",
          agent: {
            name: "Call Prep Briefer",
            schedule: "30 min before each call",
            blocks: ["Google Calendar", "Web Search", "Send Email"],
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
            title: "Call prep — Acme Corp · 2:00 PM",
            bullets: [
              "Series B, ~140 people — raised $30M in March",
              "You're meeting Dana Patel, VP Ops — second call",
              "Open thread: pricing objection on the per-seat model",
            ],
          },
        },
      },
      {
        delayMs: 600,
        part: {
          type: "text",
          text: "Done — you'll get a brief like this before every call on your calendar. 🎉",
        },
      },
    ],
  },
];
