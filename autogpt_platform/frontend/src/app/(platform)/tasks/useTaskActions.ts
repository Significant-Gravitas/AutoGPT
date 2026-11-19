import {
  getGetTaskQueryKey,
  getGetTaskRecordQueryKey,
  getListTasksQueryKey,
  useArchiveTask,
  useCancelTask,
  usePauseTaskRepeat,
  useResumeTaskRepeat,
  useRunTask,
  useStopTaskRepeat,
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
  if ("start_at" in data) {
    return data.start_at
      ? "Couldn't change the start time"
      : "Couldn't remove the start time";
  }
  return data.title ? "Couldn't rename the task" : "Couldn't change the status";
}

/** Rename, move, archive, start, stop, cancel or re-time one task, then
 *  re-read the list, the task and its record.
 *  The callbacks sit on the mutation, not the call, so they still run when
 *  the field or menu that started it has gone. */
export function useTaskActions(taskID: string) {
  const queryClient = useQueryClient();

  function refresh() {
    void queryClient.invalidateQueries({ queryKey: getListTasksQueryKey() });
    void queryClient.invalidateQueries({
      queryKey: getGetTaskQueryKey(taskID),
    });
    void queryClient.invalidateQueries({
      queryKey: getGetTaskRecordQueryKey(taskID),
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

  const runTask = useRunTask({
    mutation: {
      onSuccess: refresh,
      onError: (error) => showError("Couldn't start the task", error),
    },
  });
  const cancelTask = useCancelTask({
    mutation: {
      onSuccess: refresh,
      onError: (error) => showError("Couldn't stop the task", error),
    },
  });
  const cancelStart = useCancelTask({
    mutation: {
      onSuccess: refresh,
      onError: (error) => showError("Couldn't cancel the task", error),
    },
  });

  const pauseRepeat = usePauseTaskRepeat({
    mutation: {
      onSuccess: refresh,
      onError: (error) => showError("Couldn't pause the task", error),
    },
  });
  const resumeRepeat = useResumeTaskRepeat({
    mutation: {
      onSuccess: refresh,
      onError: (error) => showError("Couldn't resume the task", error),
    },
  });
  const stopRepeat = useStopTaskRepeat({
    mutation: {
      onSuccess: refresh,
      onError: (error) => showError("Couldn't stop it", error),
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

  /** Start, try again or run again; `instructions` say what to change. */
  function run(instructions?: string, onStarted?: () => void) {
    const text = instructions?.trim();
    runTask.mutate(
      { taskId: taskID, data: text ? { instructions: text } : null },
      { onSuccess: () => onStarted?.() },
    );
  }

  function stop(onStopped?: () => void) {
    cancelTask.mutate({ taskId: taskID }, { onSuccess: () => onStopped?.() });
  }

  /** A repeating task: stop only the run going now; it keeps repeating. */
  function stopRun(onStopped?: () => void) {
    cancelTask.mutate(
      { taskId: taskID, data: { this_run_only: true } },
      { onSuccess: () => onStopped?.() },
    );
  }

  /** Cancel a task that has not started yet; nothing of it runs. */
  function cancel() {
    cancelStart.mutate({ taskId: taskID });
  }

  /** Move a to-do task's start time; it starts by itself then. */
  function reschedule(startAt: Date, onDone?: () => void) {
    update.mutate(
      { taskId: taskID, data: { start_at: startAt } },
      { onSuccess: () => onDone?.() },
    );
  }

  /** Take the start time off, leaving a plain to-do. */
  function unschedule(onDone?: () => void) {
    update.mutate(
      { taskId: taskID, data: { start_at: null } },
      { onSuccess: () => onDone?.() },
    );
  }

  return {
    rename,
    cancel,
    stopRun,
    pause: () => pauseRepeat.mutate({ taskId: taskID }),
    resume: () => resumeRepeat.mutate({ taskId: taskID }),
    stopRepeating: () => stopRepeat.mutate({ taskId: taskID }),
    isChangingRepeat:
      pauseRepeat.isPending || resumeRepeat.isPending || stopRepeat.isPending,
    reschedule,
    unschedule,
    setStatus,
    archiveTask,
    run,
    stop,
    isStarting: runTask.isPending,
    isStopping: cancelTask.isPending,
    isCancelling: cancelStart.isPending,
    isPending: update.isPending || archive.isPending,
  };
}
