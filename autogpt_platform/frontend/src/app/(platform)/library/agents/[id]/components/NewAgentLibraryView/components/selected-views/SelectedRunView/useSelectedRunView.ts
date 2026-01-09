"use client";

import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV2GetASpecificPreset } from "@/app/api/__generated__/endpoints/presets/presets";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { okData } from "@/app/api/helpers";

export function useSelectedRunView(graphId: string, runId: string) {
  const executionQuery = useGetV1GetExecutionDetails(graphId, runId, {
    query: {
      refetchInterval: (q) => {
        const isSuccess = q.state.data?.status === 200;

        if (!isSuccess) return false;

        const status =
          q.state.data?.status === 200 ? q.state.data.data.status : undefined;

        if (!status) return false;
        if (
          status === AgentExecutionStatus.RUNNING ||
          status === AgentExecutionStatus.QUEUED ||
          status === AgentExecutionStatus.INCOMPLETE ||
          status === AgentExecutionStatus.REVIEW
        )
          return 1500;
        return false;
      },
      refetchIntervalInBackground: true,
      refetchOnWindowFocus: false,
    },
  });

  const run = okData(executionQuery.data);
  const status = executionQuery.data?.status;

  const presetId = run?.preset_id || undefined;

  const presetQuery = useGetV2GetASpecificPreset(presetId || "", {
    query: {
      enabled: !!presetId,
      select: okData,
    },
  });

  const httpError =
    status && status !== 200
      ? { status, statusText: `Request failed: ${status}` }
      : undefined;

  return {
    run,
    preset: presetQuery.data,
    isLoading: executionQuery.isLoading || presetQuery.isLoading,
    responseError: executionQuery.error || presetQuery.error,
    httpError,
  } as const;
}
