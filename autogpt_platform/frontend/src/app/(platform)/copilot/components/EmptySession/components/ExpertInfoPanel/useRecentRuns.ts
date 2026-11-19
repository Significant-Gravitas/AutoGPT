import { useGetV1ListAllExecutions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { okData } from "@/app/api/helpers";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useWorkAreaStore } from "../../../../workArea";
import {
  isLiveStatus,
  LIVE_POLL_MS,
} from "../../../WorkArea/WorkflowRunTab/runs";
import { pickRecentRuns } from "./helpers";

/** The newest workflow runs for the Profile panel: the user's own, or only
 *  the ones an expert started. Polls while one of them is still going. */
export function useRecentRuns(expertID: string | null) {
  const { user, isUserLoading } = useAuth();
  const userID = user?.id ?? null;
  const { agentInfoMap } = useLibraryAgents();
  const openWorkflowRun = useWorkAreaStore((state) => state.openWorkflowRun);

  const runsQuery = useGetV1ListAllExecutions({
    query: {
      select: (res) =>
        pickRecentRuns({ runs: okData(res) ?? [], userID, expertID }),
      enabled: Boolean(userID),
      refetchInterval: (query) =>
        pickRecentRuns({
          runs: okData(query.state.data) ?? [],
          userID,
          expertID,
        }).some((run) => isLiveStatus(run.status))
          ? LIVE_POLL_MS
          : false,
    },
  });

  const runs = (runsQuery.data ?? []).map((run) => ({
    run,
    name: agentInfoMap.get(run.graph_id)?.name ?? "Workflow",
  }));

  function openRun(run: GraphExecutionMeta, name: string) {
    openWorkflowRun({
      graphID: run.graph_id,
      graphVersion: run.graph_version,
      executionID: run.id,
      title: name,
    });
  }

  return {
    runs,
    isLoading: isUserLoading || runsQuery.isLoading,
    openRun,
  };
}
