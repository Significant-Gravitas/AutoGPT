import { Text } from "@/components/atoms/Text/Text";
import type { TaskAssignee } from "../../../data/types";
import {
  TaskAssigneeAvatar,
  assigneeLabel,
} from "../../TaskAssigneeAvatar/TaskAssigneeAvatar";

interface Props {
  assignee: TaskAssignee | null;
}

export function RowAssignee({ assignee }: Props) {
  if (!assignee) return null;
  return (
    <span className="inline-flex min-w-0 items-center gap-1.5">
      <TaskAssigneeAvatar assignee={assignee} size={16} />
      <Text as="span" variant="small" tone="quiet" className="truncate">
        {assigneeLabel(assignee)}
      </Text>
    </span>
  );
}
