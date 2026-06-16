import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";
import { useGetV2ListStoreAgents } from "@/app/api/__generated__/endpoints/store/store";
import { useGetWorkspaceStorageUsage } from "@/app/api/__generated__/endpoints/workspace/workspace";

export function useSidebarCounts() {
  const { data: libraryRes } = useGetV2ListLibraryAgents();
  const { data: storeRes } = useGetV2ListStoreAgents();
  const { data: storageRes } = useGetWorkspaceStorageUsage();

  const counts: Record<string, number | undefined> = {
    "/library":
      libraryRes?.status === 200
        ? libraryRes.data.pagination.total_items
        : undefined,
    "/marketplace":
      storeRes?.status === 200
        ? storeRes.data.pagination.total_items
        : undefined,
    "/artifacts":
      storageRes?.status === 200 ? storageRes.data.file_count : undefined,
  };

  return counts;
}
