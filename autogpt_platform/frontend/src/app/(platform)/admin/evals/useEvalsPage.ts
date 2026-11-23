"use client";

import { useState } from "react";

import {
  useGetV2GetEvalRun,
  useGetV2ListEvalRuns,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";

import { useCaseTabs } from "./useCaseTabs";

export function useEvalsPage() {
  const [pickedRunId, setPickedRunId] = useState<string | null>(null);
  const caseTabs = useCaseTabs();

  const runsQuery = useGetV2ListEvalRuns({ query: { select: okData } });
  const runs = runsQuery.data?.runs ?? [];
  const runId =
    runs.find((run) => run.id === pickedRunId)?.id ?? runs[0]?.id ?? "";

  const runQuery = useGetV2GetEvalRun(runId, {
    query: { enabled: Boolean(runId), select: okData },
  });
  const detail = runQuery.data;
  const casesById = new Map((detail?.cases ?? []).map((c) => [c.case_id, c]));

  function pickRun(id: string) {
    setPickedRunId(id);
    caseTabs.reset();
  }

  return {
    runs,
    commands: runsQuery.data?.commands ?? [],
    runId,
    detail,
    caseTabs,
    casesById,
    pickRun,
    isLoadingRuns: runsQuery.isLoading,
    runsError: runsQuery.error,
    isLoadingRun: Boolean(runId) && runQuery.isLoading,
    runError: runQuery.error ?? (runQuery.isSuccess && !detail ? true : null),
    retryRuns: runsQuery.refetch,
    retryRun: runQuery.refetch,
  };
}
