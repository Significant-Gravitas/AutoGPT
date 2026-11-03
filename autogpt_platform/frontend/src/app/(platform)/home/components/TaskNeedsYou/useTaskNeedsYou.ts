import { useTasksNeedingMe } from "@/app/(platform)/tasks/data/useTasksData";

export function useTaskNeedsYou() {
  const { data, isLoading } = useTasksNeedingMe();
  return { tasks: data ?? [], isLoading };
}
