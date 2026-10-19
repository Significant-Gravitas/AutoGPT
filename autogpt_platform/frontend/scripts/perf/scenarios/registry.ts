import type { Scenario } from "../lib/scenario";
import { builderConnectEdge } from "./builder-connect-edge";
import { builderDragRunning } from "./builder-drag-running";
import { builderRunEvents } from "./builder-run-events";
import { copilotBackgroundStream } from "./copilot-background-stream";
import { copilotIdle } from "./copilot-idle";
import { copilotStreamCode } from "./copilot-stream-code";
import { copilotStreamReply } from "./copilot-stream-reply";
import { copilotSwitchThreads } from "./copilot-switch-threads";
import { copilotType } from "./copilot-type";
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
];

export function findScenario(name: string) {
  const scenario = SCENARIOS.find((candidate) => candidate.name === name);
  if (!scenario) {
    const names = SCENARIOS.map((candidate) => candidate.name).join(", ");
    throw new Error(`Unknown scenario "${name}". Available: ${names}`);
  }
  return scenario;
}
