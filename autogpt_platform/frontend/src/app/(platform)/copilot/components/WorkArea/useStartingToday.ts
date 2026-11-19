import { pollInterval } from "@/app/(platform)/tasks/helpers";
import { startsToday } from "@/app/(platform)/tasks/schedule";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";

const SCHEDULED_LIMIT = 50;
// A repeating task leaves this list while a run goes and comes back after
// it, so the list is re-read now and then even when nothing in it moves.
const IDLE_POLL_MS = 60_000;

/** The tasks scheduled to start before today ends, soonest first, as the
 *  API orders them. */
export function useStartingToday(now: number) {
  const query = useListTasks(
    { scheduled: true, limit: SCHEDULED_LIMIT },
    {
      query: {
        select: (res) => okData(res)?.tasks ?? [],
        refetchInterval: (q) =>
          pollInterval(okData(q.state.data)?.tasks) || IDLE_POLL_MS,
      },
    },
  );
  return (query.data ?? []).filter((task) => startsToday(task, now));
}
