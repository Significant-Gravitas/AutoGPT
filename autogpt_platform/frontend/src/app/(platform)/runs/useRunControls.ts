"use client";

import { useQueryClient } from "@tanstack/react-query";

import { needsSetup } from "@/app/(platform)/copilot/components/WorkArea/WorkflowRunTab/runs";
import {
  getGetV1GetExecutionDetailsQueryKey,
  getGetV1ListGraphExecutionsQueryKey,
  usePostV1ExecuteGraphAgent,
  usePostV1StopGraphExecution,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { toast } from "@/components/molecules/Toast/use-toast";

interface Args {
  graphID: string;
  runID: string;
  agent: LibraryAgent | undefined;
  /** A new run started: open it. */
  onStarted: (runID: string) => void;
  /** The workflow asks for inputs or an account first: open its run form. */
  onSetup: () => void;
}

/** Stop a live run, or run the workflow again: at once when it asks for
 *  nothing, through its run form when it needs inputs or an account. */
export function useRunControls({
  graphID,
  runID,
  agent,
  onStarted,
  onSetup,
}: Args) {
  const queryClient = useQueryClient();
  const stopRun = usePostV1StopGraphExecution({
    mutation: {
      onSuccess: () =>
        queryClient.invalidateQueries({
          queryKey: getGetV1GetExecutionDetailsQueryKey(graphID, runID),
        }),
      onError: failed("The run could not be stopped"),
    },
  });
  const execute = usePostV1ExecuteGraphAgent({
    mutation: {
      onSuccess: (res) => {
        if (res.status !== 200) return;
        queryClient.invalidateQueries({
          queryKey: getGetV1ListGraphExecutionsQueryKey(graphID),
        });
        onStarted(res.data.id);
      },
      onError: failed("The workflow could not start"),
    },
  });

  /** False when nothing started right away (a form opened, or it cannot run
   *  from here). */
  function rerun(): boolean {
    if (!agent || agent.trigger_setup_info) return false;
    if (needsSetup(agent)) {
      onSetup();
      return false;
    }
    execute.mutate({
      graphId: agent.graph_id,
      graphVersion: agent.graph_version,
      data: { inputs: {}, credentials_inputs: {}, source: "library" },
    });
    return true;
  }

  function stop() {
    stopRun.mutate({ graphId: graphID, graphExecId: runID });
  }

  return { rerun, stop };
}

function failed(title: string) {
  return (error: unknown) =>
    toast({
      title,
      description: error instanceof Error ? error.message : undefined,
      variant: "destructive",
    });
}
