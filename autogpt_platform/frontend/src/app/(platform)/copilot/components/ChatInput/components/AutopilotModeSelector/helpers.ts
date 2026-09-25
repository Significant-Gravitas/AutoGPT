import {
  AlertDiamondIcon,
  SecurityCheckIcon,
  UserCheck01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { AutopilotMode } from "../../../../autopilotModeStore";

interface ModeOption {
  value: AutopilotMode;
  label: string;
  description: string;
  icon: IconSvgElement;
}

export const AUTOPILOT_MODE_OPTIONS: ModeOption[] = [
  {
    value: "ask_first",
    label: "Ask first",
    description: "Asks before every edit, command or outside action.",
    icon: UserCheck01Icon,
  },
  {
    value: "auto",
    label: "Auto",
    description:
      "Asks before anything leaves the platform, and about edits or commands that look risky.",
    icon: SecurityCheckIcon,
  },
  {
    value: "unsupervised",
    label: "Unsupervised",
    description: "Never asks. Every action runs without your approval.",
    icon: AlertDiamondIcon,
  },
];

export function getModeOption(mode: AutopilotMode) {
  return (
    AUTOPILOT_MODE_OPTIONS.find((option) => option.value === mode) ??
    AUTOPILOT_MODE_OPTIONS[1]
  );
}

export function isAutopilotMode(value: string): value is AutopilotMode {
  return AUTOPILOT_MODE_OPTIONS.some((option) => option.value === value);
}
