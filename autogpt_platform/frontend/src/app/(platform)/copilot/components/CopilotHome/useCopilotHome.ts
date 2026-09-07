import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  getGetBriefingsGetLatestBriefingQueryKey,
  useGetBriefingsGetLatestBriefing,
} from "@/app/api/__generated__/endpoints/briefings/briefings";
import { okData } from "@/app/api/helpers";

export function useCopilotHome() {
  const organizationId = useOrgTeamStore((state) => state.activeOrgID);
  const teamId = useOrgTeamStore((state) => state.activeTeamID);
  const isLoaded = useOrgTeamStore((state) => state.isLoaded);
  const request = getTenantRequestInit(organizationId, teamId, isLoaded);
  const { data, isLoading, isError, refetch } =
    useGetBriefingsGetLatestBriefing({
      query: {
        select: (res) => okData(res) ?? null,
        enabled: isLoaded,
        queryKey: getTeamScopedQueryKey(
          getGetBriefingsGetLatestBriefingQueryKey(),
          organizationId,
          teamId,
        ),
      },
      request,
    });

  return {
    briefing: data ?? null,
    isLoadingBriefing: isLoading,
    isBriefingError: isError,
    refetchBriefing: refetch,
  };
}
