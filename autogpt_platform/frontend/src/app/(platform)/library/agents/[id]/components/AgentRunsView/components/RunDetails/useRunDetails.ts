"use client";

import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GetV1GetExecutionDetails200 } from "@/app/api/__generated__/models/getV1GetExecutionDetails200";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";

export function useRunDetails(graphId: string, runId: string) {
  const query = useGetV1GetExecutionDetails(graphId, runId, {
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
          status === AgentExecutionStatus.INCOMPLETE
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

  const httpError =
    status && status !== 200
      ? { status, statusText: `Request failed: ${status}` }
      : undefined;

  return {
    run,
    isLoading: query.isLoading,
    responseError: query.error,
    httpError,
  } as const;
}
