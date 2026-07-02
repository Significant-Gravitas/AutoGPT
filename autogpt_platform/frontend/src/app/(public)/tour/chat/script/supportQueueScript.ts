import { RUN_DELAY_MS } from "./constants";
import type { TourScript } from "./types";

export const supportQueueScript: TourScript = [
  {
    assistantMessageId: "support-asst-1",
    userPrompt:
      "Triage new support tickets and draft replies for the common ones",
    steps: [
      {
        delayMs: 400,
        part: {
          type: "text",
          text: "On it. Here's how I'll keep your queue under control.",
        },
      },
      {
        delayMs: 700,
        part: {
          type: "plan",
          plan: {
            goal: "Triage the support queue and draft the easy replies",
            steps: [
              {
                description: "Pull new tickets from the support inbox",
                blockName: "Gmail Read",
              },
              {
                description: "Classify urgency and draft a reply",
                blockName: "AI Text Generator",
              },
              {
                description: "Escalate urgent tickets to the team",
                blockName: "Send Slack Message",
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
    assistantMessageId: "support-asst-2",
    userPrompt: "Go ahead, set it up",
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
            name: "Support Queue Triage",
            schedule: "Every 15 minutes",
            blocks: ["Gmail Read", "AI Text Generator", "Send Slack Message"],
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
            caption: "what your team sees",
            title: "Queue triaged — 14 new tickets",
            bullets: [
              "9 replies drafted and ready for your review",
              "3 billing questions routed to Sam",
              "2 urgent EU outage reports escalated to #support",
            ],
          },
        },
      },
      {
        delayMs: 600,
        part: {
          type: "text",
          text: "Running every 15 minutes — your queue stays triaged while you work. 🎉",
        },
      },
    ],
  },
];
