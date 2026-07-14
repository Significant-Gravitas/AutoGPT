import { useOrgTeamStore } from "@/services/org-team/store";
import { getQueryClient } from "@/lib/react-query/queryClient";

export function useOrgTeamSwitcher() {
  const { orgs, activeOrgID, setActiveOrg, isLoaded } = useOrgTeamStore();

  const activeOrg = orgs.find((o) => o.id === activeOrgID) || null;

  function switchOrg(orgID: string) {
    if (orgID === activeOrgID) return;
    setActiveOrg(orgID);
    // resetQueries (not clear) — clear() strands mounted observers in a
    // forever-pending state; resetQueries refetches on-screen queries
    // with the new org context.
    const queryClient = getQueryClient();
    queryClient.resetQueries();
  }

  return {
    orgs,
    activeOrg,
    switchOrg,
    isLoaded,
  };
}
