import { OTTO_VALUE } from "@/app/(platform)/tasks/components/NewTaskDialog/helpers";
import { patchTask, restore } from "@/app/(platform)/tasks/taskCache";
import { showError } from "@/app/(platform)/tasks/taskToasts";
import {
  getGetTaskQueryKey,
  getGetTaskRecordQueryKey,
  getListTasksQueryKey,
  useUpdateTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import { useQueryClient } from "@tanstack/react-query";

/** Give the task to Otto or an expert. Shows at once; put back with the
 *  server's reason when it says no (it refuses while the task is worked). */
export function useOwnerChange(task: Task) {
  const queryClient = useQueryClient();

  const update = useUpdateTask({
    mutation: {
      onMutate: ({ data }) =>
        patchTask(queryClient, task.id, (current) => ({
          ...current,
          owner_expert_id: data.assign_to_otto
            ? null
            : (data.owner_expert_id ?? current.owner_expert_id),
        })),
      onError: (error, _vars, snapshot) => {
        if (snapshot) restore(queryClient, snapshot);
        showError("Couldn't change who has it", error);
      },
      onSettled: () => {
        for (const queryKey of [
          getListTasksQueryKey(),
          getGetTaskQueryKey(task.id),
          getGetTaskRecordQueryKey(task.id),
        ]) {
          void queryClient.invalidateQueries({ queryKey });
        }
      },
    },
  });

  function pick(value: string) {
    const current = task.owner_expert_id ?? OTTO_VALUE;
    if (value === current) return;
    update.mutate({
      taskId: task.id,
      data:
        value === OTTO_VALUE
          ? { assign_to_otto: true }
          : { owner_expert_id: value },
    });
  }

  return { pick };
}
