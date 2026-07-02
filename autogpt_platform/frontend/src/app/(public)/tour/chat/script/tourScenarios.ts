import {
  HeadsetIcon,
  MagnifyingGlassIcon,
  PhoneCallIcon,
  SunIcon,
} from "@phosphor-icons/react";
import { callPrepScript } from "./callPrepScript";
import { competitorWatchScript } from "./competitorWatchScript";
import { dailyBriefScript } from "./dailyBriefScript";
import { supportQueueScript } from "./supportQueueScript";
import type { TourScenario } from "./types";

export const tourScenarios: TourScenario[] = [
  {
    id: "daily-brief",
    label: "Daily brief",
    icon: SunIcon,
    script: dailyBriefScript,
  },
  {
    id: "call-prep",
    label: "Call prep",
    icon: PhoneCallIcon,
    script: callPrepScript,
  },
  {
    id: "competitor-watch",
    label: "Competitor watch",
    icon: MagnifyingGlassIcon,
    script: competitorWatchScript,
  },
  {
    id: "support-queue",
    label: "Support queue",
    icon: HeadsetIcon,
    script: supportQueueScript,
  },
];

export const DEFAULT_SCENARIO_ID = "competitor-watch";

export function getTourScenario(id: string): TourScenario {
  return (
    tourScenarios.find((scenario) => scenario.id === id) ??
    tourScenarios.find((scenario) => scenario.id === DEFAULT_SCENARIO_ID) ??
    tourScenarios[0]
  );
}
