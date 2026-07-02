import { callPrepScript } from "./callPrepScript";
import { competitorWatchScript } from "./competitorWatchScript";
import { dailyBriefScript } from "./dailyBriefScript";
import { supportQueueScript } from "./supportQueueScript";
import type { TourScenario } from "./types";

export const tourScenarios: TourScenario[] = [
  { id: "daily-brief", label: "Daily brief", script: dailyBriefScript },
  { id: "call-prep", label: "Call prep", script: callPrepScript },
  {
    id: "competitor-watch",
    label: "Competitor watch",
    script: competitorWatchScript,
  },
  { id: "support-queue", label: "Support queue", script: supportQueueScript },
];

export const DEFAULT_SCENARIO_ID = "competitor-watch";

export function getTourScenario(id: string): TourScenario {
  return (
    tourScenarios.find((scenario) => scenario.id === id) ?? tourScenarios[0]
  );
}
