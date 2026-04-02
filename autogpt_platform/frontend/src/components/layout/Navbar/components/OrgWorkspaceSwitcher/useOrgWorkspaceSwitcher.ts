import { useOrgWorkspaceStore } from "@/services/org-workspace/store";
import { getQueryClient } from "@/lib/react-query/queryClient";

export function useOrgWorkspaceSwitcher() {
  const {
    orgs,
    workspaces,
    activeOrgID,
    activeWorkspaceID,
    setActiveOrg,
    setActiveWorkspace,
    isLoaded,
  } = useOrgWorkspaceStore();

  const activeOrg = orgs.find((o) => o.id === activeOrgID) || null;
  const activeWorkspace =
    workspaces.find((w) => w.id === activeWorkspaceID) || null;

  function switchOrg(orgID: string) {
    if (orgID === activeOrgID) return;
    setActiveOrg(orgID);
    // Clear cache to force refetch with new org context
    const queryClient = getQueryClient();
    queryClient.clear();
  }

  function switchWorkspace(workspaceID: string) {
    if (workspaceID === activeWorkspaceID) return;
    setActiveWorkspace(workspaceID);
    // Clear cache for workspace-scoped data
    const queryClient = getQueryClient();
    queryClient.clear();
  }

  return {
    orgs,
    workspaces,
    activeOrg,
    activeWorkspace,
    switchOrg,
    switchWorkspace,
    isLoaded,
  };
}
