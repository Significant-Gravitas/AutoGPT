"use client";

import { useGetV2ListGrantsSharedWithMyTeams } from "@/app/api/__generated__/endpoints/grants/grants";
import type { ReceivedGrantResponse } from "@/app/api/__generated__/models/receivedGrantResponse";
import { useOrgTeamStore } from "@/services/org-team/store";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

export function useSharedWithTeamsSection() {
  const orgId = useOrgTeamStore((s) => s.activeOrgID);
  const hasTeams = useOrgTeamStore((s) => s.teams.length > 0);
  const isEnabled = useGetFlag(Flag.SHOW_ORG_SETTINGS);

  const query = useGetV2ListGrantsSharedWithMyTeams(orgId ?? "", {
    query: {
      enabled: isEnabled && Boolean(orgId) && hasTeams,
      select: (res) => res.data as ReceivedGrantResponse[],
    },
  });

  return {
    hasTeams: isEnabled && hasTeams,
    grants: isEnabled ? (query.data ?? []) : [],
    isLoading: query.isLoading,
    isError: query.isError,
  };
}
