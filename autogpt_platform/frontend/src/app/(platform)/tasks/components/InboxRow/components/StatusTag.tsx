import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { Task } from "../../../data/types";
import { STATE_TONE, statusLabel } from "../helpers";

interface Props {
  task: Task;
}

export function StatusTag({ task }: Props) {
  const tone = task.inbox.needs_you ? "waiting" : STATE_TONE[task.inbox.state];
  return (
    <span
      className={cn(
        "inline-flex h-5 shrink-0 items-center rounded-control px-1.5",
        tone === "ok" ? "bg-status-ok-surface" : "bg-surface-wash",
      )}
    >
      <Text
        as="span"
        variant="small"
        weight="medium"
        tone={tone}
        className="whitespace-nowrap"
      >
        {statusLabel(task)}
      </Text>
    </span>
  );
}
