"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useOrgTeamStore } from "@/services/org-team/store";

interface Props {
  teamId: string | null | undefined;
  size?: "small" | "medium";
  className?: string;
}

// Small badge naming the team a row belongs to. Renders nothing for org-home
// rows (null teamId) or a team the store doesn't know about, so lists stay
// quiet for solo users and personal-org content.
export function TeamBadge({ teamId, size = "small", className }: Props) {
  const enabled = useGetFlag(Flag.SHOW_ORG_SETTINGS);
  const teams = useOrgTeamStore((s) => s.teams);

  if (!enabled || !teamId) return null;
  const team = teams.find((t) => t.id === teamId);
  if (!team) return null;

  return (
    <Badge variant="info" size={size} className={className}>
      {team.name}
    </Badge>
  );
}
