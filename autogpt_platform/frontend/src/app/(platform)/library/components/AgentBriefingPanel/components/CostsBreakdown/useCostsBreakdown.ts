import {
  getGetV1UserCostSummaryQueryKey,
  useGetV1UserCostSummary,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { UserExecutionCostSummary } from "@/app/api/__generated__/models/userExecutionCostSummary";
import { useMemo } from "react";
import { buildAgentLookup } from "./helpers";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";

export function useCostsBreakdown(
  agents: LibraryAgent[],
  { enabled }: { enabled: boolean },
) {
  const activeOrgID = useOrgTeamStore((state) => state.activeOrgID);
  const activeTeamID = useOrgTeamStore((state) => state.activeTeamID);
  const isTenantReady = useOrgTeamStore((state) => state.isLoaded);
  const {
    data: summary,
    isLoading,
    isError,
  } = useGetV1UserCostSummary(undefined, {
    query: {
      queryKey: getTeamScopedQueryKey(
        getGetV1UserCostSummaryQueryKey(),
        activeOrgID,
        activeTeamID,
      ),
      select: (res) => res.data as UserExecutionCostSummary,
      staleTime: 60_000,
      enabled: enabled && isTenantReady,
    },
    request: getTenantRequestInit(activeOrgID, activeTeamID, isTenantReady),
  });

  const agentLookup = useMemo(() => buildAgentLookup(agents), [agents]);

  const hasAnySpend = (summary?.total_cents ?? 0) > 0;

  return {
    summary,
    agentLookup,
    isLoading,
    isError,
    hasAnySpend,
  };
}
