import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  getGetHomeDashboardQueryKey,
  useGetHomeDashboard,
} from "@/app/api/__generated__/endpoints/home/home";
import { okData } from "@/app/api/helpers";

interface Args {
  enabled: boolean;
}

export function useHomePage({ enabled }: Args) {
  const organizationId = useOrgTeamStore((state) => state.activeOrgID);
  const teamId = useOrgTeamStore((state) => state.activeTeamID);
  const isLoaded = useOrgTeamStore((state) => state.isLoaded);
  const request = getTenantRequestInit(organizationId, teamId, isLoaded);
  const query = useGetHomeDashboard({
    request,
    query: {
      select: (response) => okData(response) ?? null,
      enabled: enabled && isLoaded,
      queryKey: getTeamScopedQueryKey(
        getGetHomeDashboardQueryKey(),
        organizationId,
        teamId,
      ),
      refetchInterval: 60_000,
      refetchOnWindowFocus: true,
    },
  });

  return {
    dashboard: query.data ?? null,
    isLoading: enabled && query.isLoading,
    isError: query.isError,
    refetch: query.refetch,
  };
}
