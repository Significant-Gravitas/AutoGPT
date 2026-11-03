import {
  useTaskActivity,
  useTaskDetail,
  useTaskExperts,
  useUpdateTask,
} from "../../data/useTasksData";
import type { UpdateTaskInput } from "../../data/types";

export function useTaskPanel(taskId: string) {
  const detail = useTaskDetail(taskId);
  const activity = useTaskActivity(taskId);
  const experts = useTaskExperts();
  const updateTask = useUpdateTask();

  function update(input: UpdateTaskInput) {
    updateTask.mutate({ id: taskId, input });
  }

  function retry() {
    detail.refetch();
  }

  return {
    task: detail.data,
    isLoading: detail.isLoading,
    isError: detail.isError,
    retry,
    activity: activity.data ?? [],
    isActivityLoading: activity.isLoading,
    experts: experts.data ?? [],
    update,
    isUpdating: updateTask.isPending,
  };
}
