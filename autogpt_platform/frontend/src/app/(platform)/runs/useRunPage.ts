"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import {
  isLiveStatus,
  LIVE_POLL_MS,
} from "@/app/(platform)/copilot/components/WorkArea/WorkflowRunTab/runs";
import { useRunTrace } from "@/app/(platform)/copilot/components/WorkArea/WorkflowRunTab/useRunTrace";
import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV2GetLibraryAgentByGraphId } from "@/app/api/__generated__/endpoints/library/library";
import { okData } from "@/app/api/helpers";

import { retryRunQuery, runHref, RUN_RETRY_DELAY_MS } from "./helpers";
import { useRunControls } from "./useRunControls";
import { useStepTabs } from "./useStepTabs";

/** One workflow run on a page of its own: the run, polled while it is
 *  going, its trace, the block runs open beside it, and Stop and Rerun. */
export function useRunPage(graphID: string, runID: string) {
  const router = useRouter();
  const [isSettingUp, setIsSettingUp] = useState(false);
  const tabs = useStepTabs();

  const agent = useGetV2GetLibraryAgentByGraphId(graphID, undefined, {
    query: { select: okData, staleTime: 30_000 },
  }).data;
  const runQuery = useGetV1GetExecutionDetails(graphID, runID, {
    query: {
      select: okData,
      refetchInterval: (query) => {
        const res = query.state.data;
        const status = res?.status === 200 ? res.data.status : undefined;
        return isLiveStatus(status) ? LIVE_POLL_MS : false;
      },
      refetchOnWindowFocus: false,
      retry: retryRunQuery,
      retryDelay: RUN_RETRY_DELAY_MS,
    },
  });
  const run = runQuery.data;
  const isLive = !!run && isLiveStatus(run.status);
  const { trace, now } = useRunTrace({
    graphID,
    graphVersion: run?.graph_version ?? null,
    run,
    isLive,
  });

  function openRun(nextRunID: string) {
    setIsSettingUp(false);
    tabs.reset();
    router.push(runHref(graphID, nextRunID));
  }

  const controls = useRunControls({
    graphID,
    runID,
    agent,
    onStarted: openRun,
    onSetup: () => setIsSettingUp(true),
  });

  return {
    data: {
      graphID,
      title: agent?.name || "Workflow",
      run,
      trace,
      isLive,
      now,
    },
    agent,
    tabs,
    controls,
    isSettingUp,
    cancelSetup: () => setIsSettingUp(false),
    openRun,
    notFound: runQuery.isError || (runQuery.isSuccess && !run),
    error: runQuery.error,
    retry: () => runQuery.refetch(),
  };
}
