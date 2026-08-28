"use client";

import {
  getGetV1GetExecutionDetailsQueryKey,
  useGetV1GetExecutionDetails,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV2GetASpecificPresetQueryKey,
  useGetV2GetASpecificPreset,
} from "@/app/api/__generated__/endpoints/presets/presets";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import {
  getTenantRequestInit,
  getTeamScopedQueryKey,
} from "@/components/contextual/TeamPicker/helpers";

export function useSelectedRunView(agent: LibraryAgent, runId: string) {
  const organizationId = agent.organization_id ?? null;
  const teamId = agent.team_id ?? null;
  const executionQuery = useGetV1GetExecutionDetails(agent.graph_id, runId, {
    request: getTenantRequestInit(organizationId, teamId),
    query: {
      queryKey: getTeamScopedQueryKey(
        getGetV1GetExecutionDetailsQueryKey(agent.graph_id, runId),
        organizationId,
        teamId,
      ),
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
    request: getTenantRequestInit(organizationId, teamId),
    query: {
      queryKey: getTeamScopedQueryKey(
        getGetV2GetASpecificPresetQueryKey(presetId || ""),
        organizationId,
        teamId,
      ),
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
