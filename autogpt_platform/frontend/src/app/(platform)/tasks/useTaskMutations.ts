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
import { useQueryClient } from "@tanstack/react-query";
import type { TaskPatch } from "./taskCache";
import { isAlreadyRunning } from "./taskErrors";
import { optimisticTask } from "./taskMutations";
import {
  archivedPatch,
  cancelledPatch,
  notRepeatingPatch,
  pausedPatch,
  runStoppedPatch,
  startedPatch,
  updatePatch,
} from "./taskPatches";
import { showArchived, undoArchive } from "./archiveUndo";
import { updateErrorTitle } from "./taskToasts";

/** The task's changes, each shown at once and put back on failure (see
 *  `optimisticTask`), then re-read. `useTaskActions` is what callers use. */
export function useTaskMutations(taskID: string) {
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

  function options<TVars>(
    patch: TaskPatch,
    title: string,
    retry: (vars: TVars) => void,
  ) {
    return optimisticTask<TVars>({
      queryClient,
      taskID,
      patchOf: () => patch,
      errorTitle: () => title,
      retry,
      onSettled: refresh,
    });
  }

  const update = useUpdateTask({
    mutation: optimisticTask({
      queryClient,
      taskID,
      patchOf: ({ data }) => updatePatch(data),
      errorTitle: ({ data }) => updateErrorTitle(data),
      retry: (vars) => update.mutate(vars),
      onSettled: refresh,
    }),
  });
  const archive = useArchiveTask({
    mutation: optimisticTask({
      queryClient,
      taskID,
      patchOf: () => archivedPatch,
      errorTitle: () => "Couldn't archive the task",
      retry: (vars) => archive.mutate(vars),
      onSettled: refresh,
      // The card has gone; Undo brings it back.
      afterSuccess: (before) => {
        showArchived(
          before ? () => void undoArchive(queryClient, before) : undefined,
        );
      },
    }),
  });
  const runTask = useRunTask({
    mutation: optimisticTask({
      queryClient,
      taskID,
      patchOf: () => startedPatch,
      errorTitle: () => "Couldn't start the task",
      retry: (vars) => runTask.mutate(vars),
      onSettled: refresh,
      isDone: isAlreadyRunning,
    }),
  });
  const cancelTask = useCancelTask({
    mutation: options(cancelledPatch, "Couldn't stop the task", (vars) =>
      cancelTask.mutate(vars),
    ),
  });
  const cancelRun = useCancelTask({
    mutation: options(runStoppedPatch, "Couldn't stop this run", (vars) =>
      cancelRun.mutate(vars),
    ),
  });
  const cancelStart = useCancelTask({
    mutation: options(cancelledPatch, "Couldn't cancel the task", (vars) =>
      cancelStart.mutate(vars),
    ),
  });
  const pauseRepeat = usePauseTaskRepeat({
    mutation: options(pausedPatch(true), "Couldn't pause the task", (vars) =>
      pauseRepeat.mutate(vars),
    ),
  });
  const resumeRepeat = useResumeTaskRepeat({
    mutation: options(pausedPatch(false), "Couldn't resume the task", (vars) =>
      resumeRepeat.mutate(vars),
    ),
  });
  const stopRepeat = useStopTaskRepeat({
    mutation: options(notRepeatingPatch, "Couldn't stop it", (vars) =>
      stopRepeat.mutate(vars),
    ),
  });

  return {
    update,
    archive,
    runTask,
    cancelTask,
    cancelRun,
    cancelStart,
    pauseRepeat,
    resumeRepeat,
    stopRepeat,
  };
}
