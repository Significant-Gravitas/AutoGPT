import {
  useTaskActivity,
  useTaskDetail,
  useTaskExperts,
  useUpdateTask,
} from "../../data/useTasksData";
import type { UpdateTaskInput } from "../../data/types";

export function useTaskSheet(taskId: string | null) {
  const detail = useTaskDetail(taskId);
  const activity = useTaskActivity(taskId);
  const experts = useTaskExperts();
  const updateTask = useUpdateTask();

  function update(input: UpdateTaskInput) {
    if (!taskId) return;
    updateTask.mutate({ id: taskId, input });
  }

  return {
    task: detail.data,
    isLoading: detail.isLoading,
    isError: detail.isError,
    activity: activity.data ?? [],
    isActivityLoading: activity.isLoading,
    experts: experts.data ?? [],
    update,
    isUpdating: updateTask.isPending,
  };
}
