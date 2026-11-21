import { useGetWorkflowPage } from "@/app/api/__generated__/endpoints/workflow-pages/workflow-pages";
import { okData } from "@/app/api/helpers";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

/** Whether the workflow opens on its page, so nothing should pick a run first. */
export function useHasWorkflowPage(libraryAgentID: string) {
  const enabled = Boolean(useGetFlag(Flag.WORKFLOW_PAGES));
  const query = useGetWorkflowPage(libraryAgentID, undefined, {
    query: { enabled, retry: false, select: okData },
  });
  return {
    pending: enabled && query.isLoading,
    hasPage: enabled && Boolean(query.data),
  };
}
