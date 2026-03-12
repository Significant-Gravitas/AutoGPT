"use client";

import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";

function findAgentWithLatestSuccessfulRun(
  agents: LibraryAgent[],
): LibraryAgent | null {
  for (const agent of agents) {
    const hasCompletedRun = agent.recent_executions?.some(
      (exec) => exec.status === "COMPLETED",
    );
    if (hasCompletedRun) {
      return agent;
    }
  }
  return null;
}

export function useJumpBackIn() {
  const { data, isLoading } = useGetV2ListLibraryAgents(
    {
      page: 1,
      page_size: 20,
      sort_by: "updatedAt",
    },
    {
      query: { select: okData },
    },
  );

  const agent = data ? findAgentWithLatestSuccessfulRun(data.agents) : null;

  return {
    agent,
    isLoading,
  };
}
