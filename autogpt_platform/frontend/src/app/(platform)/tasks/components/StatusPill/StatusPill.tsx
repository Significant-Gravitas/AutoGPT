import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { statusOf } from "../../helpers";

const PILL_FILL: Record<TaskStatus, string> = {
  TODO: "bg-surface-wash",
  IN_PROGRESS: "bg-info-surface",
  NEEDS_INPUT: "bg-status-waiting/10",
  FAILED: "bg-error-surface",
  DONE: "bg-status-ok-surface",
  CANCELLED: "bg-surface-wash",
};

interface Props {
  status: TaskStatus;
  className?: string;
}

/** A task's one status word on a soft tint of its colour. */
export function StatusPill({ status, className }: Props) {
  const view = statusOf(status);
  return (
    <span
      data-testid="task-status"
      data-status={status}
      className={cn(
        "inline-flex h-5 shrink-0 items-center rounded-control px-1.5",
        PILL_FILL[status],
        className,
      )}
    >
      <Text
        as="span"
        variant="inbox-medium"
        tone={view.tone}
        className="whitespace-nowrap"
      >
        {view.word}
      </Text>
    </span>
  );
}
