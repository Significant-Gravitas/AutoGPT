import {
  useGetV2GetExecutionDiagnostics,
  useGetV2GetAgentDiagnostics,
  useGetV2GetScheduleDiagnostics,
} from "@/app/api/__generated__/endpoints/admin/admin";
import type { ExecutionDiagnosticsResponse } from "@/app/api/__generated__/models/executionDiagnosticsResponse";
import type { AgentDiagnosticsResponse } from "@/app/api/__generated__/models/agentDiagnosticsResponse";
import type { ScheduleHealthMetrics } from "@/app/api/__generated__/models/scheduleHealthMetrics";

export function useDiagnosticsContent() {
  const {
    data: executionResponse,
    isLoading: isLoadingExecutions,
    isError: isExecutionError,
    error: executionError,
    refetch: refetchExecutions,
  } = useGetV2GetExecutionDiagnostics();

  const {
    data: agentResponse,
    isLoading: isLoadingAgents,
    isError: isAgentError,
    error: agentError,
    refetch: refetchAgents,
  } = useGetV2GetAgentDiagnostics();

  const {
    data: scheduleResponse,
    isLoading: isLoadingSchedules,
    isError: isScheduleError,
    error: scheduleError,
    refetch: refetchSchedules,
  } = useGetV2GetScheduleDiagnostics();

  const isLoading =
    isLoadingExecutions || isLoadingAgents || isLoadingSchedules;
  const isError = isExecutionError || isAgentError || isScheduleError;
  const error = executionError || agentError || scheduleError;

  const executionData = executionResponse?.data as
    | ExecutionDiagnosticsResponse
    | undefined;
  const agentData = agentResponse?.data as AgentDiagnosticsResponse | undefined;
  const scheduleData = scheduleResponse?.data as
    | ScheduleHealthMetrics
    | undefined;

  const refresh = () => {
    refetchExecutions();
    refetchAgents();
    refetchSchedules();
  };

  return {
    executionData,
    agentData,
    scheduleData,
    isLoading,
    isError,
    error,
    refresh,
  };
}
