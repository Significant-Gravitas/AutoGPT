import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { getExpertRoleLabel } from "@/services/experts/expert-role-label";
import type { TaskOwner } from "../helpers";

const AVATAR_SIZE = 28;

export function OwnerAvatar({ owner }: { owner: TaskOwner }) {
  if (owner.kind === "otto")
    return <AutopilotAvatar size={AVATAR_SIZE} className="shrink-0" />;
  return (
    <ExpertAvatar
      name={owner.expert.name}
      avatarUrl={owner.expert.avatarUrl}
      backgroundColor={owner.expert.color ?? undefined}
      size={AVATAR_SIZE}
      className="shrink-0"
    />
  );
}

// "{Name}, {role}" for an expert's task; "Otto" in the accent for his own.
export function OwnerLabel({ owner }: { owner: TaskOwner }) {
  if (owner.kind === "otto")
    return (
      // Text has no accent tone, so the role classes go on the span.
      <span className="font-sans text-caption font-medium text-accent-strong sm:whitespace-nowrap">
        Otto
      </span>
    );
  const role = owner.expert.role ? getExpertRoleLabel(owner.expert.role) : null;
  return (
    <Text
      variant="small"
      as="span"
      tone="secondary"
      className="sm:whitespace-nowrap"
    >
      <Text variant="small-medium" as="span" tone="primary">
        {owner.expert.name}
      </Text>
      {role ? `, ${role}` : null}
    </Text>
  );
}
