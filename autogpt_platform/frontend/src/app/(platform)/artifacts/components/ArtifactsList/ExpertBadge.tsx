"use client";

import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { Badge } from "@/components/atoms/Badge/Badge";

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
      <Badge variant="info" size="small">
        {expert.name}
      </Badge>
    </span>
  );
}
