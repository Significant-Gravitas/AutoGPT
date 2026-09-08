"use client";

import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { Badge } from "@/components/atoms/Badge/Badge";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";

interface Props {
  expertId: string | null | undefined;
  className?: string;
}

export function ExpertBadge({ expertId, className }: Props) {
  const { expertsById } = useExpertMap();
  const expert = expertId ? expertsById.get(expertId) : undefined;
  if (!expert) return null;

  return (
    <span
      className={className}
      title={`From ${expert.name}`}
      data-testid="artifacts-expert-badge"
    >
      <Badge variant="info" size="small" className="pl-1">
        <span aria-hidden className="shrink-0">
          <ExpertAvatar
            name={expert.name}
            avatarUrl={expert.avatarUrl}
            size={14}
          />
        </span>
        {expert.name}
      </Badge>
    </span>
  );
}
