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
import { useQueryClient } from "@tanstack/react-query";
import {
  patchTask,
  restore,
  type CacheSnapshot,
  type TaskPatch,
} from "./taskCache";
import {
  archivedPatch,
  cancelledPatch,
  notRepeatingPatch,
  pausedPatch,
  runStoppedPatch,
  startedPatch,
  updatePatch,
} from "./taskPatches";
import { showError, updateErrorTitle } from "./taskToasts";

/** Rename, move, archive, start, stop, cancel or re-time one task. Each
 *  change shows at once in the lists and the task's own caches, is put
 *  back with a toast if the server says no, and is re-read either way.
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

  function optimistic<TVars>(
    patchOf: (vars: TVars) => TaskPatch,
    errorTitle: (vars: TVars) => string,
  ) {
    return {
      onMutate: (vars: TVars) => patchTask(queryClient, taskID, patchOf(vars)),
      onError: (
        error: unknown,
        vars: TVars,
        snapshot: CacheSnapshot | undefined,
      ) => {
        if (snapshot) restore(queryClient, snapshot);
        showError(errorTitle(vars), error);
      },
      onSettled: refresh,
    };
  }

  const update = useUpdateTask({
    mutation: optimistic(
      ({ data }) => updatePatch(data),
      ({ data }) => updateErrorTitle(data),
    ),
  });
  const archive = useArchiveTask({
    mutation: optimistic(
      () => archivedPatch,
      () => "Couldn't archive the task",
    ),
  });
  const runTask = useRunTask({
    mutation: optimistic(
      () => startedPatch,
      () => "Couldn't start the task",
    ),
  });
  const cancelTask = useCancelTask({
    mutation: optimistic(
      () => cancelledPatch,
      () => "Couldn't stop the task",
    ),
  });
  const cancelRun = useCancelTask({
    mutation: optimistic(
      () => runStoppedPatch,
      () => "Couldn't stop this run",
    ),
  });
  const cancelStart = useCancelTask({
    mutation: optimistic(
      () => cancelledPatch,
      () => "Couldn't cancel the task",
    ),
  });
  const pauseRepeat = usePauseTaskRepeat({
    mutation: optimistic(
      () => pausedPatch(true),
      () => "Couldn't pause the task",
    ),
  });
  const resumeRepeat = useResumeTaskRepeat({
    mutation: optimistic(
      () => pausedPatch(false),
      () => "Couldn't resume the task",
    ),
  });
  const stopRepeat = useStopTaskRepeat({
    mutation: optimistic(
      () => notRepeatingPatch,
      () => "Couldn't stop it",
    ),
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
    cancelRun.mutate(
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
    /** Stop: the whole task. */
    isStopping: cancelTask.isPending,
    /** Stop this run: a repeating task's run going now. */
    isStoppingRun: cancelRun.isPending,
    isCancelling: cancelStart.isPending,
    isPending: update.isPending || archive.isPending,
  };
}
