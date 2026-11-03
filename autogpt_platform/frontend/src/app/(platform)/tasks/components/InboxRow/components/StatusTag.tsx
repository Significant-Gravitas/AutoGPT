import { Text } from "@/components/atoms/Text/Text";
import type { Task } from "../../../data/types";
import { STATE_TONE, statusLabel } from "../helpers";

interface Props {
  task: Task;
}

export function StatusTag({ task }: Props) {
  const tone = task.inbox.needs_you ? "waiting" : STATE_TONE[task.inbox.state];
  return (
    <span className="inline-flex shrink-0 items-center rounded-full bg-foreground-ink/5 px-1.5">
      <Text
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
