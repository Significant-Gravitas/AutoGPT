import {
  getGetV2ListTurnsAtBudgetResetQueryKey,
  useGetV2ListTurnsAtBudgetReset,
  type getV2ListTurnsAtBudgetResetResponse,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { AtResetTurn } from "@/app/api/__generated__/models/atResetTurn";
import { useQueryClient } from "@tanstack/react-query";

/** The chat's turns set for the budget reset, and a way to update them
 *  without waiting for the next read. */
export function useAtResetJobs(sessionId: string | null) {
  const queryClient = useQueryClient();
  const { data: jobs } = useGetV2ListTurnsAtBudgetReset(sessionId ?? "", {
    query: {
      enabled: !!sessionId,
      select: (response) => (response.status === 200 ? response.data.jobs : []),
      // A set turn disappears once it fires; look again now and then.
      refetchInterval: (query) =>
        query.state.data?.status === 200 && query.state.data.data.jobs.length
          ? 60_000
          : false,
    },
  });

  function writeJobs(update: (jobs: AtResetTurn[]) => AtResetTurn[]) {
    queryClient.setQueryData<getV2ListTurnsAtBudgetResetResponse>(
      getGetV2ListTurnsAtBudgetResetQueryKey(sessionId ?? ""),
      (old) => ({
        status: 200,
        headers: old?.headers ?? new Headers(),
        data: { jobs: update(old?.status === 200 ? old.data.jobs : []) },
      }),
    );
  }

  return { jobs, writeJobs };
}
