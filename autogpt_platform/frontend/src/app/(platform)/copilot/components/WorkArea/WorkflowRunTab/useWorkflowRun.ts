import {
  useGetV1GetExecutionDetails,
  useGetV1ListGraphExecutions,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV2GetLibraryAgentByGraphId } from "@/app/api/__generated__/endpoints/library/library";
import { okData } from "@/app/api/helpers";
import { useEffect } from "react";
import { useWorkAreaStore, type WorkTab } from "../../../workArea";
import { isLiveStatus, LIVE_POLL_MS } from "./runs";
import { useRunTrace } from "./useRunTrace";

export type WorkflowTab = Extract<WorkTab, { kind: "workflow" }>;

const RUNS_PAGE_SIZE = 20;

/** A workflow's tab: the workflow and its runs newest first, or, once a run
 *  is picked, that run and its trace, polled while it is going. */
export function useWorkflowRun(tab: WorkflowTab) {
  const openWorkflowRun = useWorkAreaStore((s) => s.openWorkflowRun);
  const nameWorkflow = useWorkAreaStore((s) => s.nameWorkflow);
  const agent = useGetV2GetLibraryAgentByGraphId(tab.graphID, undefined, {
    query: { select: okData, staleTime: 30_000 },
  }).data;
  const name = agent?.name;
  useEffect(() => {
    if (name) nameWorkflow(tab.graphID, name);
  }, [tab.graphID, name, nameWorkflow]);

  const runsQuery = useGetV1ListGraphExecutions(
    tab.graphID,
    { page: 1, page_size: RUNS_PAGE_SIZE },
    {
      query: {
        select: okData,
        refetchInterval: (query) => {
          const page = query.state.data;
          const anyLive =
            page?.status === 200 &&
            page.data.executions.some((run) => isLiveStatus(run.status));
          return anyLive ? LIVE_POLL_MS * 2 : false;
        },
      },
    },
  );
  const runs = runsQuery.data?.executions ?? [];
  const selectedID = tab.executionID;

  const runQuery = useGetV1GetExecutionDetails(tab.graphID, selectedID ?? "", {
    query: {
      enabled: !!selectedID,
      select: okData,
      refetchInterval: (query) => {
        const res = query.state.data;
        const status = res?.status === 200 ? res.data.status : undefined;
        return isLiveStatus(status) ? LIVE_POLL_MS : false;
      },
      refetchOnWindowFocus: false,
    },
  });
  const run = runQuery.data;
  const isLive = !!run && isLiveStatus(run.status);
  // A run that cannot be read (deleted, or an id that was never a run) has
  // nothing to show: the tab goes back to the workflow's list of runs
  // instead of waiting on it for ever.
  const isRunMissing =
    !!selectedID && !run && (runQuery.isError || runQuery.isSuccess);
  const graphID = tab.graphID;
  const tabTitle = tab.title;
  useEffect(() => {
    if (!isRunMissing) return;
    openWorkflowRun({ graphID, executionID: null, title: tabTitle });
  }, [isRunMissing, graphID, tabTitle, openWorkflowRun]);

  const { trace, now } = useRunTrace({
    graphID: tab.graphID,
    graphVersion: run?.graph_version ?? tab.graphVersion,
    run,
    isLive,
  });

  function showRun(executionID: string) {
    openWorkflowRun({ graphID: tab.graphID, executionID, title: tab.title });
  }

  function showRuns() {
    openWorkflowRun({
      graphID: tab.graphID,
      executionID: null,
      title: tab.title,
    });
  }

  return {
    agent,
    title: agent?.name || tab.title || "Workflow",
    runs,
    isLoadingRuns: runsQuery.isLoading,
    selectedID,
    run,
    isLive,
    now,
    trace,
    showRun,
    showRuns,
  };
}
