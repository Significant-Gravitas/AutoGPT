import { useOrgTeamStore } from "@/services/org-team/store";
import { getQueryClient } from "@/lib/react-query/queryClient";

export function useOrgTeamSwitcher() {
  const {
    orgs,
    teams,
    activeOrgID,
    activeTeamID,
    setActiveOrg,
    setActiveTeam,
    isLoaded,
  } = useOrgTeamStore();

  const activeOrg = orgs.find((o) => o.id === activeOrgID) || null;
  const activeTeam = teams.find((w) => w.id === activeTeamID) || null;

  function switchOrg(orgID: string) {
    if (orgID === activeOrgID) return;
    setActiveOrg(orgID);
    // Clear cache to force refetch with new org context
    const queryClient = getQueryClient();
    queryClient.clear();
  }

  function switchTeam(teamID: string) {
    if (teamID === activeTeamID) return;
    setActiveTeam(teamID);
    // Clear cache for team-scoped data
    const queryClient = getQueryClient();
    queryClient.clear();
  }

  return {
    orgs,
    teams,
    activeOrg,
    activeTeam,
    switchOrg,
    switchTeam,
    isLoaded,
  };
}
