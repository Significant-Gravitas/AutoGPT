import { useGetV1UserCostSummary } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { UserExecutionCostSummary } from "@/app/api/__generated__/models/userExecutionCostSummary";
import { useMemo } from "react";
import { buildAgentLookup } from "./helpers";

export function useCostsBreakdown(agents: LibraryAgent[]) {
  const {
    data: summary,
    isLoading,
    isError,
  } = useGetV1UserCostSummary(undefined, {
    query: {
      select: (res) => res.data as UserExecutionCostSummary,
      staleTime: 60_000,
    },
  });

  const agentLookup = useMemo(() => buildAgentLookup(agents), [agents]);

  const hasAnySpend = (summary?.total_cents ?? 0) > 0;

  return {
    summary,
    agentLookup,
    isLoading,
    isError,
    hasAnySpend,
  };
}
