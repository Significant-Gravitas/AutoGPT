import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { statusOf } from "../../data/helpers";
import type { TaskState } from "../../data/types";
import { StateDot } from "../StateDot/StateDot";

interface Props {
  state: TaskState;
  className?: string;
}

/** A task's one status mark: a dot and its word, red when it failed. */
export function StatusBadge({ state, className }: Props) {
  const status = statusOf(state);
  return (
    <span
      data-testid="task-status"
      className={cn("inline-flex shrink-0 items-center gap-1", className)}
    >
      <StateDot state={state} decorative className="-ml-1" />
      <Text
        as="span"
        variant="inbox-medium"
        tone={status.tone}
        className="whitespace-nowrap"
      >
        {status.word}
      </Text>
    </span>
  );
}
