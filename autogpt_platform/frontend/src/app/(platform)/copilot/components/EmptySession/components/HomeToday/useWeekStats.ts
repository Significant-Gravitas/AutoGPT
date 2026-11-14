import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";
import { useSessionList } from "@/app/(platform)/copilot/useSessionList";
import { buildWeekStats } from "./weekStats";

export function useWeekStats(now: number) {
  const tasksQuery = useListTasks(
    { limit: 100 },
    { query: { select: (res) => okData(res)?.tasks ?? [] } },
  );
  const { sessions } = useSessionList();
  return buildWeekStats(tasksQuery.data ?? [], sessions, now);
}
