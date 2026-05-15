import { useGetV1UserCostSummary } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { UserExecutionCostSummary } from "@/app/api/__generated__/models/userExecutionCostSummary";
import { useMemo } from "react";
import { buildAgentLookup, fillDailyGaps } from "./helpers";

export function useCostsBreakdown(agents: LibraryAgent[]) {
  const {
    data: rawSummary,
    isLoading,
    isError,
  } = useGetV1UserCostSummary(undefined, {
    query: {
      select: (res) => res.data as UserExecutionCostSummary,
      staleTime: 60_000,
    },
  });

  const agentLookup = useMemo(() => buildAgentLookup(agents), [agents]);

  const summary = useMemo(() => {
    if (!rawSummary) return rawSummary;
    return { ...rawSummary, daily: fillDailyGaps(rawSummary.daily) };
  }, [rawSummary]);

  const hasAnySpend = (summary?.total_cents ?? 0) > 0;

  return {
    summary,
    agentLookup,
    isLoading,
    isError,
    hasAnySpend,
  };
}
