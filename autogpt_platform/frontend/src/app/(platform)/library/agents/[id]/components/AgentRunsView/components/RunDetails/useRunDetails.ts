"use client";

import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GetV1GetExecutionDetails200 } from "@/app/api/__generated__/models/getV1GetExecutionDetails200";

export function useRunDetails(graphId: string, runId: string) {
  const query = useGetV1GetExecutionDetails(graphId, runId);

  const status = query.data?.status;

  const run: GetV1GetExecutionDetails200 | undefined =
    status === 200
      ? (query.data?.data as GetV1GetExecutionDetails200)
      : undefined;

  const httpError =
    status && status !== 200
      ? { status, statusText: "Request failed" }
      : undefined;

  return {
    run,
    isLoading: query.isLoading,
    error: query.error,
    httpError,
  } as const;
}
