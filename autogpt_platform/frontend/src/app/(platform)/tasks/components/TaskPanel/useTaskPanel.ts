import {
  useTaskDetail,
  useTaskExperts,
  useTaskSummary,
  useUpdateTask,
} from "../../data/useTasksData";
import type { UpdateTaskInput } from "../../data/types";
import { panelView } from "./panelView";

export function useTaskPanel(taskId: string) {
  const detail = useTaskDetail(taskId);
  const summary = useTaskSummary(taskId);
  const experts = useTaskExperts();
  const updateTask = useUpdateTask();

  function update(input: UpdateTaskInput) {
    updateTask.mutate({ id: taskId, input });
  }

  function retry() {
    detail.refetch();
    summary.refetch();
  }

  return {
    task: detail.data,
    view: panelView(detail.data, summary.data),
    isLoading: detail.isLoading,
    isError: detail.isError,
    retry,
    experts: experts.data ?? [],
    update,
  };
}
