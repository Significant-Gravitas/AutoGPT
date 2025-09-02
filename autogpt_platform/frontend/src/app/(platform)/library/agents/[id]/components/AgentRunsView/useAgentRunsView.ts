import { useQuery } from "@tanstack/react-query";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useParams } from "next/navigation";

export function useAgentRunsView() {
  const { id } = useParams();
  const agentId = id as string;
  const api = useBackendAPI();
  
  const { data: response, isSuccess, error } = useQuery({
    queryKey: ["v2", "get", "library", "agent", agentId],
    queryFn: () => api.getLibraryAgent(agentId),
    enabled: !!agentId,
  });

  return {
    agentId: id,
    ready: isSuccess,
    error,
    response,
  };
}
