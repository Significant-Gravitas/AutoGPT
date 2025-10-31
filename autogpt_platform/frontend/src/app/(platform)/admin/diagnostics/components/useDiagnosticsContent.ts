import {
  useGetV2GetExecutionDiagnostics,
  useGetV2GetAgentDiagnostics,
} from "@/app/api/__generated__/endpoints/admin/admin";
import type { ExecutionDiagnosticsResponse } from "@/app/api/__generated__/models/executionDiagnosticsResponse";
import type { AgentDiagnosticsResponse } from "@/app/api/__generated__/models/agentDiagnosticsResponse";

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

  const isLoading = isLoadingExecutions || isLoadingAgents;
  const isError = isExecutionError || isAgentError;
  const error = executionError || agentError;

  const executionData: ExecutionDiagnosticsResponse | undefined =
    executionResponse?.data;
  const agentData: AgentDiagnosticsResponse | undefined = agentResponse?.data;

  const refresh = () => {
    refetchExecutions();
    refetchAgents();
  };

  return {
    executionData,
    agentData,
    isLoading,
    isError,
    error,
    refresh,
  };
}
