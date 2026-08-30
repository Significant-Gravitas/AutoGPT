import { getListExpertRunsQueryOptions } from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertRun } from "@/app/api/__generated__/models/expertRun";
import { okData } from "@/app/api/helpers";
import { useQueries } from "@tanstack/react-query";
import { useState } from "react";
import { sortTasksByRecency } from "./helpers";

interface Args {
  experts: Expert[];
  enabled: boolean;
}

/** There is no cross-expert runs endpoint, so the board fans out one
 *  `list_expert_runs` query per hired expert and merges the results. */
export function useAllTasksSection({ experts, enabled }: Args) {
  const [needsReviewOnly, setNeedsReviewOnly] = useState(false);

  const results = useQueries({
    queries: experts.map((expert) =>
      getListExpertRunsQueryOptions<ExpertRun[]>(expert.id, {
        query: { select: (res) => okData(res) ?? [], enabled },
      }),
    ),
  });

  const tasks = sortTasksByRecency(
    results.flatMap((result, index) =>
      (result.data ?? []).map((run) => ({ run, expert: experts[index] })),
    ),
  );
  const reviewCount = tasks.filter((task) => task.run.needs_review).length;

  return {
    tasks: needsReviewOnly
      ? tasks.filter((task) => task.run.needs_review)
      : tasks,
    reviewCount,
    needsReviewOnly,
    toggleNeedsReviewOnly: () => setNeedsReviewOnly((value) => !value),
    isLoading: results.some((result) => result.isLoading),
    // Only a total failure is worth replacing the list with an error card; a
    // single expert erroring still leaves real tasks to show.
    isError: results.length > 0 && results.every((result) => result.isError),
    refetch: () => results.forEach((result) => void result.refetch()),
  };
}
