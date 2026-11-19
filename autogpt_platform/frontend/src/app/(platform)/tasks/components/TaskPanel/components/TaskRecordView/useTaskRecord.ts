import { useGetTaskRecord } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { isSettled, POLL_MS } from "../../../../helpers";

/** The task's record, refreshed while the task is still moving. */
export function useTaskRecord(taskID: string) {
  const query = useGetTaskRecord(taskID, {
    query: {
      select: (res) => (res.status === 200 ? res.data : null),
      refetchInterval: (q) => {
        const res = q.state.data;
        if (res?.status !== 200) return false;
        return isSettled(res.data.task.status) ? false : POLL_MS;
      },
    },
  });
  return { record: query.data ?? null };
}
