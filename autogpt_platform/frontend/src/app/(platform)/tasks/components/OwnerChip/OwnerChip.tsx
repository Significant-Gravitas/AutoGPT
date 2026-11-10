import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import type { TaskOwner } from "../../helpers";

interface Props {
  owner: TaskOwner;
  size?: number;
}

/** Who works the task: their avatar and name. */
export function OwnerChip({ owner, size = 16 }: Props) {
  return (
    <span
      data-testid="task-owner"
      className="inline-flex min-w-0 items-center gap-1.5"
    >
      <span aria-hidden className="inline-flex shrink-0">
        <ExpertAvatar
          name={owner.name}
          avatarUrl={owner.avatarUrl}
          size={size}
        />
      </span>
      <Text as="span" variant="small" tone="quiet" className="truncate">
        {owner.name}
      </Text>
    </span>
  );
}
