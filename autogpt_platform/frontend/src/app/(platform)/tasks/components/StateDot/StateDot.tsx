import { cn } from "@/lib/utils";
import { STATE_LABELS } from "../../data/helpers";
import type { TaskState } from "../../data/types";

const DOT: Record<TaskState, string> = {
  working: "size-2 bg-info animate-pulse motion-reduce:animate-none",
  needs_you: "size-2 bg-status-waiting",
  stuck: "size-1.5 bg-warning",
  done: "size-2 bg-status-ok",
  stopped: "size-2 bg-foreground-faint",
};

interface Props {
  state: TaskState;
  className?: string;
}

export function StateDot({ state, className }: Props) {
  return (
    <span
      role="img"
      aria-label={STATE_LABELS[state]}
      className={cn(
        "inline-flex size-4 shrink-0 items-center justify-center",
        className,
      )}
    >
      <span aria-hidden className={cn("rounded-full", DOT[state])} />
    </span>
  );
}
