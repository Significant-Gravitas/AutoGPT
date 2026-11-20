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
      // A set turn disappears once it fires; look again now and then, as
      // one comes due, and often while it is, to show it went (or was moved
      // on).
      refetchInterval: (query) =>
        query.state.data?.status === 200
          ? refetchEvery(query.state.data.data.jobs)
          : false,
    },
  });

  async function writeJobs(update: (jobs: AtResetTurn[]) => AtResetTurn[]) {
    const queryKey = getGetV2ListTurnsAtBudgetResetQueryKey(sessionId ?? "");
    // A read already on its way would land after this and put the old
    // list back.
    await queryClient.cancelQueries({ queryKey });
    queryClient.setQueryData<getV2ListTurnsAtBudgetResetResponse>(
      queryKey,
      (old) => ({
        status: 200,
        headers: old?.headers ?? new Headers(),
        data: { jobs: update(old?.status === 200 ? old.data.jobs : []) },
      }),
    );
  }

  return { jobs, writeJobs };
}

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;

function refetchEvery(jobs: AtResetTurn[]) {
  if (!jobs.length) return false;
  const untilDue =
    Math.min(...jobs.map((job) => new Date(job.run_at).getTime())) - Date.now();
  if (untilDue <= 0) return 10_000;
  // Changes made here update the list directly, and it's read again on
  // focus, so a turn hours away only needs an occasional look.
  const ceiling = untilDue > HOUR ? 15 * MINUTE : MINUTE;
  return Math.min(Math.max(untilDue, 1_000), ceiling);
}
