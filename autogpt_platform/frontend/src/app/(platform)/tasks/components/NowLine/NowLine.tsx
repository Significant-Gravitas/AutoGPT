import type { Task } from "@/app/api/__generated__/models/task";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { nowLineOf, subtaskProgressOf } from "../../nowLine";

interface Props {
  task: Task;
  className?: string;
}

/** One quiet line under a task's title: what it is doing now, what came of
 *  it or why it failed, and how many of its subtasks are done. */
export function NowLine({ task, className }: Props) {
  const line = nowLineOf(task);
  const progress = subtaskProgressOf(task);
  if (!line && !progress) return null;

  return (
    <span
      data-testid="task-now"
      className={cn("flex min-w-0 items-center gap-2", className)}
    >
      {line ? (
        <Text
          as="span"
          variant="small"
          tone="quiet"
          className="min-w-0 truncate"
          title={line}
          dir="auto"
        >
          {line}
        </Text>
      ) : null}
      {progress ? (
        <Text
          as="span"
          variant="small"
          tone="faint"
          tabular
          className="ml-auto shrink-0"
        >
          <span aria-hidden>
            {progress.done} / {progress.total}
          </span>
          <span className="sr-only">
            {progress.done} of {progress.total} subtasks done
          </span>
        </Text>
      ) : null}
    </span>
  );
}
