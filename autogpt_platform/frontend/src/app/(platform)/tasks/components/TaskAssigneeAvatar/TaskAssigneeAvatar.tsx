import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import type { TaskAssignee } from "../../data/types";

interface Props {
  assignee: TaskAssignee | null;
  size?: number;
  withName?: boolean;
}

export function assigneeLabel(assignee: TaskAssignee | null) {
  if (!assignee) return "Unassigned";
  if (assignee.type === "user") return "Me";
  if (assignee.type === "otto") return "Otto";
  return assignee.name;
}

export function TaskAssigneeAvatar({
  assignee,
  size = 20,
  withName = false,
}: Props) {
  const label = assigneeLabel(assignee);
  if (!assignee) {
    return withName ? (
      <Text variant="small" tone="muted">
        {label}
      </Text>
    ) : null;
  }
  return (
    <span className="inline-flex min-w-0 items-center gap-1.5" title={label}>
      <ExpertAvatar
        name={label}
        avatarUrl={assignee.avatar_url ?? null}
        size={size}
      />
      {withName ? (
        <Text variant="small" tone="body" className="truncate">
          {label}
        </Text>
      ) : null}
    </span>
  );
}
