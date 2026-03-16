"use client";

import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";
import { okData } from "@/app/api/helpers";

export function useJumpBackIn() {
  const { data, isLoading } = useGetV2ListLibraryAgents(
    {
      page: 1,
      page_size: 1,
      sort_by: "updatedAt",
    },
    {
      query: { select: okData },
    },
  );

  // The API doesn't include execution data by default (include_executions is
  // internal to the backend), so recent_executions is always empty here.
  // We use the most recently updated agent as the "jump back in" candidate
  // instead — updatedAt is the best available proxy for recent activity.
  const agent = data?.agents[0] ?? null;

  return {
    agent,
    isLoading,
  };
}
