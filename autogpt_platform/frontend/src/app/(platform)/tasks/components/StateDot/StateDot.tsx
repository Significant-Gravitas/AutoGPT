import { cn } from "@/lib/utils";
import { statusOf } from "../../data/helpers";
import type { TaskState } from "../../data/types";

const DOT: Record<TaskState, string> = {
  working: "bg-info animate-pulse motion-reduce:animate-none",
  needs_you: "bg-status-waiting",
  stuck: "bg-error",
  done: "bg-status-ok",
  stopped: "bg-foreground-faint",
};

interface Props {
  state: TaskState;
  className?: string;
  /** Beside a visible status word, the dot is decoration only. */
  decorative?: boolean;
}

export function StateDot({ state, className, decorative = false }: Props) {
  return (
    <span
      role={decorative ? undefined : "img"}
      aria-label={decorative ? undefined : statusOf(state).word}
      aria-hidden={decorative || undefined}
      className={cn(
        "inline-flex size-4 shrink-0 items-center justify-center",
        className,
      )}
    >
      <span aria-hidden className={cn("size-2 rounded-full", DOT[state])} />
    </span>
  );
}
