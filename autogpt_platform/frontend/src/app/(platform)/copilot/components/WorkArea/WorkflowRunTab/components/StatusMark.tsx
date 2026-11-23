import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  Loading03Icon,
  StopCircleIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import type { StepState } from "../trace";
import styles from "../trace.module.css";

export const STATE_LABELS: Record<StepState, string> = {
  waiting: "Waiting",
  running: "Running",
  done: "Done",
  failed: "Failed",
  stopped: "Stopped",
};

interface Props {
  state: StepState;
}

/** A block run's small mark: a dashed ring while it waits, a turning ring
 *  while it runs, a check, a cross, or a stop sign once it settles. The word
 *  rides along for screen readers. */
export function StatusMark({ state }: Props) {
  return (
    <span
      data-state={state}
      className="inline-flex size-4 shrink-0 items-center justify-center"
    >
      <Glyph state={state} />
      <span className="sr-only">{STATE_LABELS[state]}</span>
    </span>
  );
}

function Glyph({ state }: Props) {
  switch (state) {
    case "running":
      return (
        <Icon icon={Loading03Icon} size={13} tone="info" spin aria-hidden />
      );
    case "done":
      return <Icon icon={Tick02Icon} size={13} tone="ok" aria-hidden />;
    case "failed":
      return <Icon icon={Cancel01Icon} size={13} tone="error" aria-hidden />;
    case "stopped":
      return <Icon icon={StopCircleIcon} size={13} tone="faint" aria-hidden />;
    default:
      return (
        <span
          aria-hidden
          className={cn("size-2.5 rounded-full", styles.waitingRing)}
        />
      );
  }
}
