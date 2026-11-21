import {
  getGetWorkflowPageQueryKey,
  useDeleteWorkflowPage,
  useGetWorkflowPage,
  useRemoveWorkflowPageSection,
  useRestoreWorkflowPage,
} from "@/app/api/__generated__/endpoints/workflow-pages/workflow-pages";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useQueryClient } from "@tanstack/react-query";
import { isRunActive } from "./helpers";
import { useFollowedRunStore } from "./store";

const POLL_MS = 2_000;

export function useWorkflowPageMount(agent: LibraryAgent) {
  const enabled = Boolean(useGetFlag(Flag.WORKFLOW_PAGES));
  const runID = useFollowedRunStore((s) => s.runs[agent.id]);
  const follow = useFollowedRunStore((s) => s.follow);
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const query = useGetWorkflowPage(
    agent.id,
    runID ? { run_id: runID } : undefined,
    {
      query: {
        enabled,
        retry: false,
        select: okData,
        refetchInterval: (q) =>
          isRunActive(followedStatus(okData(q.state.data))) ? POLL_MS : false,
      },
    },
  );

  function refresh() {
    return queryClient.invalidateQueries({
      queryKey: getGetWorkflowPageQueryKey(agent.id),
    });
  }

  const mutation = {
    onSuccess: refresh,
    onError: () =>
      toast({ title: "Could not change the page", variant: "destructive" }),
  };
  const removeSection = useRemoveWorkflowPageSection({ mutation });
  const restore = useRestoreWorkflowPage({ mutation });
  const reset = useDeleteWorkflowPage({ mutation });

  return {
    enabled,
    isLoading: enabled && query.isLoading,
    page: query.data,
    onRunStarted: (run: { id: string }) => follow(agent.id, run.id),
    onRemoveSection: (index: number) =>
      removeSection.mutate({ libraryAgentId: agent.id, index }),
    onRestore: () => restore.mutate({ libraryAgentId: agent.id }),
    onReset: () => reset.mutate({ libraryAgentId: agent.id }),
  };
}

function followedStatus(page: { sections: { kind: string }[] } | undefined) {
  const result = page?.sections.find((s) => s.kind === "latest_result");
  return result && "run_status" in result
    ? (result.run_status as string)
    : null;
}
