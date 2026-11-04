import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import type { TaskAssignee } from "../../../../data/types";

interface Props {
  assignee: TaskAssignee | null;
  size?: number;
}

export function AssigneeAvatar({ assignee, size = 20 }: Props) {
  return (
    <ExpertAvatar
      name={assignee?.name ?? "Unassigned"}
      avatarUrl={assignee?.avatar_url ?? null}
      size={size}
    />
  );
}
