"use client";

import { useGetV2ListGrantsSharedWithMyTeams } from "@/app/api/__generated__/endpoints/grants/grants";
import type { ReceivedGrantResponse } from "@/app/api/__generated__/models/receivedGrantResponse";
import { useOrgTeamStore } from "@/services/org-team/store";

export function useSharedWithTeamsSection() {
  const orgId = useOrgTeamStore((s) => s.activeOrgID);
  const hasTeams = useOrgTeamStore((s) => s.teams.length > 0);

  const query = useGetV2ListGrantsSharedWithMyTeams(orgId ?? "", {
    query: {
      enabled: Boolean(orgId) && hasTeams,
      select: (res) => res.data as ReceivedGrantResponse[],
    },
  });

  return {
    hasTeams,
    grants: query.data ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
  };
}
