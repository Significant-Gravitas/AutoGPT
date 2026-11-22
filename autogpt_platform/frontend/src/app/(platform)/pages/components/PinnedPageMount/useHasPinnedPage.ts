import { useGetPinnedPage } from "@/app/api/__generated__/endpoints/pages/pages";
import { okData } from "@/app/api/helpers";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

/** Whether the workflow opens on a pinned page, so nothing should pick a run first. */
export function useHasPinnedPage(libraryAgentID: string) {
  const enabled = Boolean(useGetFlag(Flag.PAGES));
  const query = useGetPinnedPage(libraryAgentID, undefined, {
    query: { enabled, retry: false, select: okData },
  });
  return {
    pending: enabled && query.isLoading,
    hasPage: enabled && Boolean(query.data),
  };
}
