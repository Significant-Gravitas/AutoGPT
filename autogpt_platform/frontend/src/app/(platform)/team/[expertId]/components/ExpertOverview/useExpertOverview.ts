"use client";

import { useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { ExpertActivity } from "@/app/api/__generated__/models/expertActivity";
import { okData } from "@/app/api/helpers";
import { isInternalToolOutput } from "@/app/(platform)/copilot/components/ContextPanel/components/FilesTab/helpers";
import { isHiddenWorkspaceFile } from "@/lib/workspace-files";
import { getChartDays, getWindowStat } from "./helpers";

export const RECENT_CONVERSATIONS_LIMIT = 5;
const OUTPUTS_LIMIT = 200;

export function useExpertOverview(
  expertId: string,
  activity: ExpertActivity | null,
) {
  const sessionsQuery = useGetV2ListSessions(
    { expert_id: expertId, limit: RECENT_CONVERSATIONS_LIMIT },
    { query: { select: okData } },
  );
  const filesQuery = useListWorkspaceFiles(
    { expert_id: expertId, origin: "generated", limit: OUTPUTS_LIMIT },
    { query: { select: okData } },
  );

  const days = activity?.days ?? [];
  const outputs = (filesQuery.data?.files ?? []).filter(
    (file) => !isInternalToolOutput(file) && !isHiddenWorkspaceFile(file),
  );

  return {
    chartDays: getChartDays(days),
    hasActivity: days.some((day) => day.sessions > 0 || day.runs > 0),
    conversations: getWindowStat(days, "sessions"),
    runs: getWindowStat(days, "runs"),
    sessions: sessionsQuery.data?.sessions ?? [],
    isSessionsLoading: sessionsQuery.isLoading,
    isSessionsError: sessionsQuery.isError,
    retrySessions: () => sessionsQuery.refetch(),
    outputs,
    outputCount: outputs.length,
    hasMoreOutputs: Boolean(filesQuery.data?.has_more),
    isOutputsLoading: filesQuery.isLoading,
    isOutputsError: filesQuery.isError,
    retryOutputs: () => filesQuery.refetch(),
  };
}
