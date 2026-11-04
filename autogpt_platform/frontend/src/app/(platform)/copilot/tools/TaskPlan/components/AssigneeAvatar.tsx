import type { TaskAssignee } from "@/app/(platform)/tasks/data/types";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";

interface Props {
  assignee: TaskAssignee;
  size?: number;
}

export function AssigneeAvatar({ assignee, size = 20 }: Props) {
  if (assignee.type === "otto") return <AutopilotAvatar size={size} />;
  return (
    <ExpertAvatar
      name={assignee.name}
      avatarUrl={assignee.avatar_url ?? null}
      size={size}
    />
  );
}
