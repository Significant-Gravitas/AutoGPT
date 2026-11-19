import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { useOneAtATime } from "./useOneAtATime";
import { useTaskMutations } from "./useTaskMutations";

/** Rename, move, archive, start, stop, cancel or re-time one task. Each
 *  change shows at once in the lists and the task's own caches, is put
 *  back with one toast if it fails, and is re-read either way (see
 *  `optimisticTask`). Start, stop, cancel and archive run one at a time,
 *  so a double click sends one request; a start the server says is
 *  already running counts as done.
 *  The callbacks sit on the mutation, not the call, so they still run when
 *  the field or menu that started it has gone. */
export function useTaskActions(taskID: string) {
  const once = useOneAtATime();
  const {
    update,
    archive,
    runTask,
    cancelTask,
    cancelRun,
    cancelStart,
    pauseRepeat,
    resumeRepeat,
    stopRepeat,
  } = useTaskMutations(taskID);

  function rename(title: string) {
    update.mutate({ taskId: taskID, data: { title } });
  }

  function setStatus(status: TaskStatus) {
    update.mutate({ taskId: taskID, data: { status } });
  }

  function archiveTask(onArchived?: () => void) {
    once("archive", (done) =>
      archive.mutate(
        { taskId: taskID },
        { onSuccess: () => onArchived?.(), onSettled: done },
      ),
    );
  }

  /** Start, try again or run again; `instructions` say what to change. */
  function run(instructions?: string, onStarted?: () => void) {
    const text = instructions?.trim();
    once("run", (done) =>
      runTask.mutate(
        { taskId: taskID, data: text ? { instructions: text } : null },
        { onSuccess: () => onStarted?.(), onSettled: done },
      ),
    );
  }

  function stop(onStopped?: () => void) {
    once("stop", (done) =>
      cancelTask.mutate(
        { taskId: taskID },
        { onSuccess: () => onStopped?.(), onSettled: done },
      ),
    );
  }

  /** A repeating task: stop only the run going now; it keeps repeating. */
  function stopRun(onStopped?: () => void) {
    once("stopRun", (done) =>
      cancelRun.mutate(
        { taskId: taskID, data: { this_run_only: true } },
        { onSuccess: () => onStopped?.(), onSettled: done },
      ),
    );
  }

  /** Cancel a task that has not started yet; nothing of it runs. */
  function cancel() {
    once("cancel", (done) =>
      cancelStart.mutate({ taskId: taskID }, { onSettled: done }),
    );
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
