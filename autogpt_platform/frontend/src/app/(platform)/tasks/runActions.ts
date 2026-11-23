import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import {
  PlayIcon,
  RepeatIcon,
  StopIcon,
  ArrowReloadHorizontalIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

/** What the user can do to a task's work from the task itself. */
export type RunAction = "start" | "stop" | "retry" | "rerun";

export interface RunActionView {
  label: string;
  icon: IconSvgElement;
}

export const RUN_ACTIONS: Record<RunAction, RunActionView> = {
  start: { label: "Start", icon: PlayIcon },
  stop: { label: "Stop", icon: StopIcon },
  retry: { label: "Try again", icon: ArrowReloadHorizontalIcon },
  rerun: { label: "Run again", icon: RepeatIcon },
};

export function runActionOf(status: TaskStatus): RunAction {
  switch (status) {
    case TaskStatus.IN_PROGRESS:
    case TaskStatus.NEEDS_INPUT:
      return "stop";
    case TaskStatus.FAILED:
    case TaskStatus.CANCELLED:
      return "retry";
    case TaskStatus.DONE:
      return "rerun";
    default:
      return "start";
  }
}
