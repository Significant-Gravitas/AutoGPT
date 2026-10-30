"use client";

import { useGetV2ListSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { ExpertActivity } from "@/app/api/__generated__/models/expertActivity";
import { okData } from "@/app/api/helpers";
import { useRaiseStatus } from "@/services/experts/raise-expert";
import { isInternalToolOutput } from "@/app/(platform)/copilot/components/ContextPanel/components/FilesTab/helpers";
import { isHiddenWorkspaceFile } from "@/lib/workspace-files";
import { getChartDays, getWindowStat } from "./helpers";

export const RECENT_CONVERSATIONS_LIMIT = 5;
const OUTPUTS_LIMIT = 200;
const EMPTY_SESSIONS = {
  status: 200 as const,
  data: { sessions: [], total: 0 },
  headers: new Headers(),
};
const EMPTY_FILES = {
  status: 200 as const,
  data: { files: [], has_more: false },
  headers: new Headers(),
};

export function useExpertOverview(
  expertId: string,
  activity: ExpertActivity | null,
) {
  // An expert raised in this tab has no conversations or outputs yet: its
  // empty states show at once while the real lists load behind them.
  const isNew = useRaiseStatus(expertId) !== undefined;
  const sessionsQuery = useGetV2ListSessions(
    { expert_id: expertId, limit: RECENT_CONVERSATIONS_LIMIT },
    {
      query: {
        select: okData,
        placeholderData: isNew ? EMPTY_SESSIONS : undefined,
      },
    },
  );
  const filesQuery = useListWorkspaceFiles(
    { expert_id: expertId, origin: "generated", limit: OUTPUTS_LIMIT },
    {
      query: {
        select: okData,
        placeholderData: isNew ? EMPTY_FILES : undefined,
      },
    },
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
