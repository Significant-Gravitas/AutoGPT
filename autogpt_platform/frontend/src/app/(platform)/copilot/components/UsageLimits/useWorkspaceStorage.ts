import { useGetWorkspaceStorageUsage } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { StorageUsageResponse } from "@/app/api/__generated__/models/storageUsageResponse";

export function useWorkspaceStorage() {
  return useGetWorkspaceStorageUsage({
    query: {
      select: (res) => res.data as StorageUsageResponse,
      staleTime: 30000,
      refetchInterval: 60000,
    },
  });
}
