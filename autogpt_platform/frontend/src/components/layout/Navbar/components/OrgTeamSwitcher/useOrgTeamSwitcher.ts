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
    // resetQueries (not clear) — clear() strands mounted observers in a
    // forever-pending state; resetQueries refetches on-screen queries
    // with the new org context.
    const queryClient = getQueryClient();
    queryClient.resetQueries();
  }

  function switchTeam(teamID: string) {
    if (teamID === activeTeamID) return;
    setActiveTeam(teamID);
    const queryClient = getQueryClient();
    queryClient.resetQueries();
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
