import type { Scenario } from "../lib/scenario";
import { builderBlockMenuSearch } from "./builder-block-menu-search";
import { builderConnectEdge } from "./builder-connect-edge";
import { builderDragRunning } from "./builder-drag-running";
import { builderLoadGraph } from "./builder-load-graph";
import { builderRunEvents } from "./builder-run-events";
import { builderRunBeads } from "./builder-run-beads";
import { builderTypeField } from "./builder-type-field";
import { copilotBackgroundStream } from "./copilot-background-stream";
import { copilotIdle } from "./copilot-idle";
import { copilotOpenLongThread } from "./copilot-open-long-thread";
import { copilotResizeDrag } from "./copilot-resize-drag";
import { copilotStreamCode } from "./copilot-stream-code";
import { copilotStreamReply } from "./copilot-stream-reply";
import { copilotSwitchThreads } from "./copilot-switch-threads";
import { copilotToolWait } from "./copilot-tool-wait";
import { copilotType } from "./copilot-type";
import { libraryAgentRuns } from "./library-agent-runs";
import { libraryRunning } from "./library-running";
import { libraryRunningEvents } from "./library-running-events";

export const SCENARIOS: Scenario[] = [
  copilotIdle,
  copilotStreamReply,
  copilotBackgroundStream,
  copilotType,
  copilotSwitchThreads,
  builderDragRunning,
  libraryRunning,
  copilotStreamCode,
  builderConnectEdge,
  libraryRunningEvents,
  builderRunEvents,
  copilotToolWait,
  copilotResizeDrag,
  builderRunBeads,
  libraryAgentRuns,
  builderTypeField,
  builderBlockMenuSearch,
  builderLoadGraph,
  copilotOpenLongThread,
];

export function findScenario(name: string) {
  const scenario = SCENARIOS.find((candidate) => candidate.name === name);
  if (!scenario) {
    const names = SCENARIOS.map((candidate) => candidate.name).join(", ");
    throw new Error(`Unknown scenario "${name}". Available: ${names}`);
  }
  return scenario;
}
