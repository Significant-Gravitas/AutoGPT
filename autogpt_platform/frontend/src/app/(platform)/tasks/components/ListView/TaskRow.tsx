import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import {
  TableCell,
  TableRow,
} from "@/components/molecules/Table/components/TablePrimitives";
import { formatMetrics, STATUS_LABELS, taskKey } from "../../data/helpers";
import type { Task } from "../../data/types";
import { DueChip } from "../DueChip/DueChip";
import { PriorityGlyph } from "../PriorityGlyph/PriorityGlyph";
import { SubtaskRing } from "../SubtaskRing/SubtaskRing";
import { TaskAssigneeAvatar } from "../TaskAssigneeAvatar/TaskAssigneeAvatar";
import { statusBadgeVariant } from "./helpers";

interface Props {
  task: Task;
  onOpen: (id: string) => void;
}

export function TaskRow({ task, onOpen }: Props) {
  return (
    <TableRow className="cursor-pointer" onClick={() => onOpen(task.id)}>
      <TableCell className="w-20">
        <Text variant="small-mono" tone="muted">
          {taskKey(task.number)}
        </Text>
      </TableCell>
      <TableCell>
        <div className="flex min-w-0 items-center gap-2">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onOpen(task.id);
            }}
            className="min-w-0 rounded-sm text-left focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none"
          >
            <Text variant="body-medium" as="span" className="block truncate">
              {task.title}
            </Text>
          </button>
          <SubtaskRing
            done={task.subtask_counts.done}
            total={task.subtask_counts.total}
          />
        </div>
      </TableCell>
      <TableCell>
        <TaskAssigneeAvatar assignee={task.assignee} withName />
      </TableCell>
      <TableCell>
        <PriorityGlyph priority={task.priority} />
      </TableCell>
      <TableCell>
        {task.due_date ? (
          <DueChip due={task.due_date} isDone={task.status === "done"} />
        ) : (
          <Text variant="small" tone="subtle">
            —
          </Text>
        )}
      </TableCell>
      <TableCell>
        <Badge variant={statusBadgeVariant(task.status)} size="small">
          {STATUS_LABELS[task.status]}
        </Badge>
      </TableCell>
      <TableCell className="hidden lg:table-cell">
        <Text variant="small" tone="secondary" className="truncate">
          {task.metrics.tool_calls > 0 ? formatMetrics(task.metrics) : "—"}
        </Text>
      </TableCell>
    </TableRow>
  );
}
