import { useTasksNeedingMe } from "@/app/(platform)/tasks/data/useTasksData";

export function useTaskNeedsYou() {
  const { data, isLoading } = useTasksNeedingMe();
  const tasks = (data ?? []).filter(
    (t) => t.state === "needs_you" || t.state === "stuck",
  );
  return { tasks, isLoading };
}
