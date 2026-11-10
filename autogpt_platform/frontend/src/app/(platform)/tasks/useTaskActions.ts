import {
  getGetTaskQueryKey,
  getListTasksQueryKey,
  useArchiveTask,
  useUpdateTask,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import type { TaskUpdateRequest } from "@/app/api/__generated__/models/taskUpdateRequest";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";

function showError(title: string, error: unknown) {
  toast({
    title,
    description: error instanceof Error ? error.message : undefined,
    variant: "destructive",
  });
}

function updateErrorTitle(data: TaskUpdateRequest) {
  return data.title ? "Couldn't rename the task" : "Couldn't change the status";
}

/** Rename, move or archive one task, then re-read the list and the task.
 *  The callbacks sit on the mutation, not the call, so they still run when
 *  the field or menu that started it has gone. */
export function useTaskActions(taskID: string) {
  const queryClient = useQueryClient();

  function refresh() {
    void queryClient.invalidateQueries({ queryKey: getListTasksQueryKey() });
    void queryClient.invalidateQueries({
      queryKey: getGetTaskQueryKey(taskID),
    });
  }

  const update = useUpdateTask({
    mutation: {
      onSuccess: refresh,
      onError: (error, { data }) => showError(updateErrorTitle(data), error),
    },
  });
  const archive = useArchiveTask({
    mutation: {
      onSuccess: refresh,
      onError: (error) => showError("Couldn't archive the task", error),
    },
  });

  function rename(title: string) {
    update.mutate({ taskId: taskID, data: { title } });
  }

  function setStatus(status: TaskStatus) {
    update.mutate({ taskId: taskID, data: { status } });
  }

  function archiveTask(onArchived?: () => void) {
    archive.mutate({ taskId: taskID }, { onSuccess: () => onArchived?.() });
  }

  return {
    rename,
    setStatus,
    archiveTask,
    isPending: update.isPending || archive.isPending,
  };
}
