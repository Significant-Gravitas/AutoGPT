"use client";

import { useGetV2ListWorkspaceMembers } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { TeamMemberResponse } from "@/app/api/__generated__/models/teamMemberResponse";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { TEAM_HEADER_NAME } from "@/services/org-team/headers";

interface Args {
  orgId: string;
  wsId: string;
}

export function useTeamMembersPreview({ orgId, wsId }: Args) {
  // Same hook (and query key) the manage panel uses, so an already-fetched
  // roster is served from cache. Only mounted once the row is expanded, so the
  // fetch is lazy. The private-team gate (403/404) lands on the team-routes
  // chain; treat it as "you can't inspect this team" rather than a hard error.
  const membersQuery = useGetV2ListWorkspaceMembers(orgId, wsId, {
    query: {
      enabled: Boolean(orgId && wsId),
      select: (res) => res.data as TeamMemberResponse[],
      retry: false,
    },
    request: { headers: { [TEAM_HEADER_NAME]: wsId } },
  });

  const error: unknown = membersQuery.error;
  const status = error instanceof ApiError ? error.status : undefined;
  const isPrivate = status === 403 || status === 404;

  return {
    members: membersQuery.data ?? [],
    isLoading: membersQuery.isLoading,
    isError: membersQuery.isError,
    isPrivate,
  };
}
