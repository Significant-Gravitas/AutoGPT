"use client";

import { useGetV2PerTeamSpendBreakdown } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgSpendResponse } from "@/app/api/__generated__/models/orgSpendResponse";

export function useTeamSpendSection(orgId: string, enabled: boolean) {
  const query = useGetV2PerTeamSpendBreakdown(orgId, undefined, {
    query: {
      enabled: enabled && Boolean(orgId),
      select: (res) => res.data as OrgSpendResponse,
    },
  });

  return {
    buckets: query.data?.teams ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    refetch: query.refetch,
  };
}
