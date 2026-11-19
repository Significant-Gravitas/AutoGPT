import { useTaskActions } from "@/app/(platform)/tasks/useTaskActions";
import { RUN_ACTIONS, runActionOf } from "@/app/(platform)/tasks/runActions";
import { scheduleStateOf } from "@/app/(platform)/tasks/schedule";
import type { Task } from "@/app/api/__generated__/models/task";
import { useState } from "react";

type Mode = "idle" | "retry" | "confirm-stop";

export function useTaskControls(task: Task) {
  const actions = useTaskActions(task.id);
  const [mode, setMode] = useState<Mode>("idle");
  const action = runActionOf(task.status);

  function done() {
    setMode("idle");
  }

  function onAction() {
    if (action === "stop") return setMode("confirm-stop");
    if (action === "retry") return setMode(mode === "retry" ? "idle" : "retry");
    actions.run(undefined, done);
  }

  return {
    action,
    /** Waiting for its start time, or missed it: Start now and Cancel. */
    schedule: scheduleStateOf(task),
    view: RUN_ACTIONS[action],
    isBusy: actions.isStarting || actions.isStopping || actions.isStoppingRun,
    isStarting: actions.isStarting,
    // A repeating task's stop ends this run only, which has its own flag.
    isStopping: task.repeat ? actions.isStoppingRun : actions.isStopping,
    isAskingRetry: mode === "retry",
    isConfirmingStop: mode === "confirm-stop",
    onAction,
    retry: (instructions: string) => actions.run(instructions, done),
    // A repeating task's Stop ends only the run going now.
    confirmStop: () =>
      task.repeat ? actions.stopRun(done) : actions.stop(done),
    startNow: () => actions.run(undefined, done),
    cancel: actions.cancel,
    isCancelling: actions.isCancelling,
    pause: actions.pause,
    resume: actions.resume,
    stopRepeating: actions.stopRepeating,
    isChangingRepeat:
      actions.isStarting || actions.isCancelling || actions.isChangingRepeat,
    done,
  };
}
