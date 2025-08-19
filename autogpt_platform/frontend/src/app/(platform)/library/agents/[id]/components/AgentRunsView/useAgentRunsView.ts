import { useGetV2GetLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { useParams } from "next/navigation";

export function useAgentRunsView() {
  const { id } = useParams();
  const agentId = id as string;
  const { data: response, isSuccess, error } = useGetV2GetLibraryAgent(agentId);

  return {
    agentId: id,
    ready: isSuccess,
    error,
    response,
  };
}
