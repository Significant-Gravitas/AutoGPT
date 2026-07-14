"use client";

import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";
import Avatar, { AvatarFallback } from "@/components/atoms/Avatar/Avatar";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";

import { useTeamMembersPreview } from "./useTeamMembersPreview";

interface Props {
  orgId: string;
  team: TeamResponse;
}

export function TeamMembersPreview({ orgId, team }: Props) {
  const { members, isLoading, isError, isPrivate } = useTeamMembersPreview({
    orgId,
    wsId: team.id,
  });

  return (
    <div
      id={`team-members-${team.id}`}
      className="rounded-lg bg-zinc-50 px-3 py-2"
      data-testid="team-members-preview"
    >
      {isLoading ? (
        <ul className="flex flex-col divide-y divide-zinc-100">
          {[0, 1].map((i) => (
            <li
              key={i}
              className="flex items-center gap-3 py-2"
              data-testid="team-member-skeleton"
            >
              <Skeleton className="size-8 shrink-0 rounded-full" />
              <div className="flex flex-1 flex-col gap-1.5">
                <Skeleton className="h-3 w-32" />
                <Skeleton className="h-3 w-44" />
              </div>
            </li>
          ))}
        </ul>
      ) : isError ? (
        <Text
          variant="small"
          className="text-zinc-500"
          data-testid="team-members-hint"
        >
          {isPrivate
            ? "Private — join this team to see its members."
            : "Couldn't load members."}
        </Text>
      ) : members.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          This team has no members yet.
        </Text>
      ) : (
        <ul className="flex flex-col divide-y divide-zinc-100">
          {members.map((member) => (
            <li
              key={member.user_id}
              className="flex items-center gap-3 py-2"
              data-testid="team-member-preview-row"
            >
              <Avatar className="size-8 shrink-0">
                <AvatarFallback className="text-xs">
                  {(member.name || member.email).charAt(0).toUpperCase()}
                </AvatarFallback>
              </Avatar>
              <div className="flex min-w-0 flex-1 flex-col">
                <span className="truncate text-sm font-medium">
                  {member.name || member.email}
                </span>
                <span className="truncate text-xs text-zinc-500">
                  {member.email}
                </span>
              </div>
              {member.is_admin ? <Badge variant="info">Admin</Badge> : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
