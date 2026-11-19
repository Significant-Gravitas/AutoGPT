"use client";

import { StopTaskDialog } from "@/app/(platform)/tasks/components/StopTaskDialog/StopTaskDialog";
import { TryAgainForm } from "@/app/(platform)/tasks/components/TryAgainForm/TryAgainForm";
import { isBetweenRuns } from "@/app/(platform)/tasks/schedule";
import type { Task } from "@/app/api/__generated__/models/task";
import { Button } from "@/components/atoms/Button/Button";
import { RepeatControls } from "./RepeatControls";
import { ScheduledControls } from "./ScheduledControls";
import { useTaskControls } from "./useTaskControls";

interface Props {
  task: Task;
}

/** The task tab's header: start, stop or run the task again; Start now
 *  or Cancel while it waits for its start time. */
export function TaskControls({ task }: Props) {
  const controls = useTaskControls(task);

  if (isBetweenRuns(controls.schedule)) {
    return (
      <RepeatControls
        title={task.title}
        isPaused={controls.schedule === "paused"}
        isWatching={!!task.repeat?.watch}
        isStarting={controls.isStarting}
        isBusy={controls.isChangingRepeat}
        onRunNow={controls.startNow}
        onPause={controls.pause}
        onResume={controls.resume}
        onStopRepeating={controls.stopRepeating}
        onCancel={controls.cancel}
      />
    );
  }
  if (controls.schedule) {
    return (
      <ScheduledControls
        isMissed={controls.schedule === "missed"}
        isStarting={controls.isStarting}
        isCancelling={controls.isCancelling}
        onStartNow={controls.startNow}
        onCancel={controls.cancel}
      />
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Button
          variant={controls.action === "stop" ? "secondary" : "primary"}
          size="xs"
          leadingIcon={controls.view.icon}
          loading={controls.isBusy}
          aria-expanded={
            controls.action === "retry" ? controls.isAskingRetry : undefined
          }
          onClick={controls.onAction}
        >
          {controls.view.label}
        </Button>
      </div>
      {controls.isAskingRetry ? (
        <TryAgainForm
          taskID={task.id}
          isPending={controls.isStarting}
          onSubmit={controls.retry}
          onCancel={controls.done}
        />
      ) : null}
      <StopTaskDialog
        isOpen={controls.isConfirmingStop}
        title={task.title}
        isPending={controls.isStopping}
        onCancel={controls.done}
        onConfirm={controls.confirmStop}
        isRepeating={!!task.repeat}
      />
    </div>
  );
}
