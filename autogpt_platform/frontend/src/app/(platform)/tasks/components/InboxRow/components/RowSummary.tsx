import { Text } from "@/components/atoms/Text/Text";
import { taskKey } from "../../../data/helpers";
import type { Task } from "../../../data/types";
import { secondaryLine } from "../helpers";
import { RowAssignee } from "./RowAssignee";
import { RowClock } from "./RowClock";
import { StatusGlyph } from "./StatusGlyph";
import { StatusTag } from "./StatusTag";
import { ToolHints } from "./ToolHints";

interface Props {
  task: Task;
  elapsed: number | null;
  variant: "list" | "board";
}

export function RowSummary({ task, elapsed, variant }: Props) {
  const running = task.inbox.state === "running";
  const line = secondaryLine(task);
  const key = (
    <Text as="span" variant="small" tone="faint" mono className="shrink-0">
      {taskKey(task.number)}
    </Text>
  );
  const clock = <RowClock elapsed={elapsed} running={running} />;

  if (variant === "board") {
    return (
      <span className="flex min-w-0 flex-col gap-1.5">
        <span className="flex min-w-0 items-start gap-2">
          <span className="pt-0.5">
            <StatusGlyph state={task.inbox.state} />
          </span>
          <Text
            as="span"
            variant="inbox-medium"
            tone="ink"
            className="line-clamp-2 min-w-0 flex-1 break-words"
          >
            {task.title}
          </Text>
        </span>
        {line ? (
          <Text as="span" variant="inbox" tone="quiet" className="truncate">
            {line}
          </Text>
        ) : null}
        <span className="flex min-w-0 items-center gap-2">
          <span className="flex min-w-0 flex-1 overflow-hidden">
            <RowAssignee assignee={task.assignee} />
          </span>
          {key}
        </span>
        <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
          <StatusTag task={task} />
          <span className="ml-auto shrink-0 tabular-nums">{clock}</span>
        </span>
      </span>
    );
  }

  return (
    <span className="flex min-w-0 items-start gap-3">
      <span className="pt-0.5">
        <StatusGlyph state={task.inbox.state} />
      </span>
      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="flex min-w-0 items-center gap-2">
          <Text
            as="span"
            variant="inbox-medium"
            tone="ink"
            className="min-w-0 truncate"
          >
            {task.title}
          </Text>
          {key}
          <StatusTag task={task} />
        </span>
        {line ? (
          <Text
            as="span"
            variant="inbox"
            tone={task.inbox.needs_you ? "text" : "quiet"}
            className="truncate"
          >
            {line}
          </Text>
        ) : null}
      </span>
      <span className="flex shrink-0 items-center gap-4 pt-0.5">
        <span className="hidden lg:inline-flex">
          <ToolHints tools={task.inbox.tools_used} />
        </span>
        <span className="hidden max-w-40 md:inline-flex">
          <RowAssignee assignee={task.assignee} />
        </span>
        {clock}
      </span>
    </span>
  );
}
