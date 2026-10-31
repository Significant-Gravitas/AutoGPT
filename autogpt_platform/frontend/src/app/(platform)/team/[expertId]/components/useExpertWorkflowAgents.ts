import { getGetV2GetLibraryAgentsByIdsQueryOptions } from "@/app/api/__generated__/endpoints/library/library";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { useQueries } from "@tanstack/react-query";
import { chunkLibraryAgentIDs } from "../../helpers";

export function useExpertWorkflowAgents(workflows: ExpertWorkflowRef[]) {
  const chunks = chunkLibraryAgentIDs(workflows);
  const results = useQueries({
    queries: chunks.map((ids) =>
      getGetV2GetLibraryAgentsByIdsQueryOptions(
        { ids },
        { query: { select: (res) => okData(res) ?? [] } },
      ),
    ),
  });

  const agentsByID = new Map<string, LibraryAgent>();
  for (const result of results) {
    for (const agent of result.data ?? []) agentsByID.set(agent.id, agent);
  }
  return agentsByID;
}
