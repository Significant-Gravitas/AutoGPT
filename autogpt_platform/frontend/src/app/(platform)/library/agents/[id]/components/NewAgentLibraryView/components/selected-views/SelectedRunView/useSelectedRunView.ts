"use client";

import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { useGetV2GetASpecificPreset } from "@/app/api/__generated__/endpoints/presets/presets";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GetV1GetExecutionDetails200 } from "@/app/api/__generated__/models/getV1GetExecutionDetails200";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { okData } from "@/app/api/helpers";

export function useSelectedRunView(graphId: string, runId: string) {
  const query = useGetV1GetExecutionDetails(graphId, runId, {
    query: {
      refetchInterval: (q: any) => {
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

  const status = query.data?.status;

  const run: GetV1GetExecutionDetails200 | undefined =
    status === 200
      ? (query.data?.data as GetV1GetExecutionDetails200)
      : undefined;

  const presetId =
    run && "preset_id" in run && run.preset_id
      ? (run.preset_id as string)
      : undefined;

  const presetQuery = useGetV2GetASpecificPreset(presetId || "", {
    query: {
      enabled: !!presetId,
      select: (res) => okData<LibraryAgentPreset>(res),
    },
  });

  const httpError =
    status && status !== 200
      ? { status, statusText: `Request failed: ${status}` }
      : undefined;

  return {
    run,
    preset: presetQuery.data,
    isLoading: query.isLoading || presetQuery.isLoading,
    responseError: query.error || presetQuery.error,
    httpError,
  } as const;
}
