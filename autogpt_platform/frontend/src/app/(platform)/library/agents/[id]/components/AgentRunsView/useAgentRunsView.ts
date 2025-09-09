import { useGetV2GetLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { useParams } from "next/navigation";
import { parseAsString, useQueryState } from "nuqs";

export function useAgentRunsView() {
  const { id } = useParams();
  const agentId = id as string;
  const { data: response, isSuccess, error } = useGetV2GetLibraryAgent(agentId);

  const [runParam, setRunParam] = useQueryState("executionId", parseAsString);
  const selectedRun = runParam ?? undefined;

  function handleSelectRun(id: string) {
    setRunParam(id, { shallow: true });
  }

  function clearSelectedRun() {
    setRunParam(null, { shallow: true });
  }

  return {
    agentId: id,
    ready: isSuccess,
    error,
    response,
    selectedRun,
    handleSelectRun,
    clearSelectedRun,
  };
}
