import {
  getGetV2ListLibraryAgentsQueryKey,
  useGetV2ListLibraryAgentsInfinite,
} from "@/app/api/__generated__/endpoints/library/library";
import { getPaginationNextPageNumber, unpaginate } from "@/app/api/helpers";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";
import { useMemo } from "react";
import { buildAgentInfoMap } from "./store";

export function useLibraryAgents() {
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const activeTeamID = useOrgTeamStore((s) => s.activeTeamID);
  const isTenantReady = useOrgTeamStore((s) => s.isLoaded);
  const params = {
    page: 1,
    page_size: 100,
    is_hidden: false,
  };
  const { data: agentsQueryData, isLoading: isRefreshing } =
    useGetV2ListLibraryAgentsInfinite(params, {
      query: {
        enabled: isTenantReady,
        getNextPageParam: getPaginationNextPageNumber,
        queryKey: getTeamScopedQueryKey(
          getGetV2ListLibraryAgentsQueryKey(params),
          activeOrgID,
          activeTeamID,
        ),
        // Don't block rendering - fetch in background
        refetchOnMount: false,
        refetchOnWindowFocus: false,
        staleTime: 5 * 60 * 1000, // 5 minutes
      },
      request: getTenantRequestInit(activeOrgID, activeTeamID, isTenantReady),
    });

  const agents = agentsQueryData ? unpaginate(agentsQueryData, "agents") : [];

  // Use agents.length as dependency to avoid recreating map unnecessarily
  const agentInfoMap = useMemo(
    () => buildAgentInfoMap(agents),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [agents.length, agents.map((a) => a.id).join(",")],
  );

  return { agents, agentInfoMap, isRefreshing, lastUpdatedAt: undefined };
}
