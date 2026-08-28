import {
  getGetWorkspaceStorageUsageQueryKey,
  useGetWorkspaceStorageUsage,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { StorageUsageResponse } from "@/app/api/__generated__/models/storageUsageResponse";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useCopilotTenantScope } from "../../CopilotTenantScopeContext";

export function useWorkspaceStorage() {
  const scope = useCopilotTenantScope();
  return useGetWorkspaceStorageUsage({
    request: getTenantRequestInit(scope.organizationId, scope.teamId),
    query: {
      queryKey: getTeamScopedQueryKey(
        getGetWorkspaceStorageUsageQueryKey(),
        scope.organizationId,
        scope.teamId,
      ),
      select: (res) => res.data as StorageUsageResponse,
      staleTime: 30000,
      refetchInterval: 60000,
    },
  });
}
