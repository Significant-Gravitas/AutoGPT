import { callPrepScript } from "./callPrepScript";
import {
  callPrepArtifact,
  competitorWatchArtifact,
  dailyBriefArtifact,
  supportQueueArtifact,
} from "./completionArtifacts";
import { competitorWatchScript } from "./competitorWatchScript";
import { dailyBriefScript } from "./dailyBriefScript";
import { supportQueueScript } from "./supportQueueScript";
import type { TourScenario } from "./types";
import {
  CallRinging01Icon,
  HeadsetIcon,
  Search01Icon,
  Sun01Icon,
} from "@hugeicons/core-free-icons";

export const tourScenarios: TourScenario[] = [
  {
    id: "daily-brief",
    label: "Daily brief",
    icon: Sun01Icon,
    script: dailyBriefScript,
    completionArtifact: dailyBriefArtifact,
  },
  {
    id: "call-prep",
    label: "Call prep",
    icon: CallRinging01Icon,
    script: callPrepScript,
    completionArtifact: callPrepArtifact,
  },
  {
    id: "competitor-watch",
    label: "Competitor watch",
    icon: Search01Icon,
    script: competitorWatchScript,
    completionArtifact: competitorWatchArtifact,
  },
  {
    id: "support-queue",
    label: "Support queue",
    icon: HeadsetIcon,
    script: supportQueueScript,
    completionArtifact: supportQueueArtifact,
  },
];

export const DEFAULT_SCENARIO_ID = "competitor-watch";

/** The scenario the end-of-demo nudge points at: the next unwatched one in
 * sidebar order (wrapping around), or simply the next one once all are
 * watched — "Watch another scenario" should always lead somewhere. */
export function getNextTourScenario(
  activeScenarioId: string,
  watchedScenarioIds: string[],
): TourScenario {
  const activeIndex = tourScenarios.findIndex(
    (scenario) => scenario.id === activeScenarioId,
  );
  const others = [
    ...tourScenarios.slice(activeIndex + 1),
    ...tourScenarios.slice(0, Math.max(activeIndex, 0)),
  ];
  return (
    others.find((scenario) => !watchedScenarioIds.includes(scenario.id)) ??
    others[0] ??
    tourScenarios[0]
  );
}

export function getTourScenario(id: string): TourScenario {
  return (
    tourScenarios.find((scenario) => scenario.id === id) ??
    tourScenarios.find((scenario) => scenario.id === DEFAULT_SCENARIO_ID) ??
    tourScenarios[0]
  );
}
