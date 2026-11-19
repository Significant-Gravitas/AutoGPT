import type { Task } from "@/app/api/__generated__/models/task";
import { useRef, useState } from "react";
import { settableStatuses } from "../../helpers";
import { RUN_ACTIONS, runActionOf } from "../../runActions";
import { copyTaskLink } from "../../taskLink";
import { useTaskActions } from "../../useTaskActions";

interface Args {
  task: Task;
  onRename: () => void;
  onArchived?: () => void;
}

export function useTaskMenu({ task, onRename, onArchived }: Args) {
  const actions = useTaskActions(task.id);
  const [isConfirmingArchive, setIsConfirmingArchive] = useState(false);
  const [runDialog, setRunDialog] = useState<"stop" | "retry" | null>(null);
  const [isDuplicating, setDuplicating] = useState(false);
  const runAction = runActionOf(task.status);
  const statuses = settableStatuses(task.status);
  // Rename moves focus into the title field; the closing menu mustn't take
  // it back to its trigger.
  const renamePicked = useRef(false);

  function startRename() {
    renamePicked.current = true;
    onRename();
  }

  function onCloseAutoFocus(e: Event) {
    if (!renamePicked.current) return;
    renamePicked.current = false;
    e.preventDefault();
  }

  function pickStatus(value: string) {
    const status = statuses.find((s) => s === value);
    if (!status || status === task.status) return;
    actions.setStatus(status);
  }

  function pickRun() {
    if (runAction === "stop" || runAction === "retry") {
      setRunDialog(runAction);
      return;
    }
    actions.run();
  }

  function closeRunDialog() {
    setRunDialog(null);
  }

  function confirmArchive() {
    actions.archiveTask(() => {
      setIsConfirmingArchive(false);
      onArchived?.();
    });
  }

  return {
    isDuplicating,
    setDuplicating,
    startDuplicate: () => setDuplicating(true),
    copyLink: () => void copyTaskLink(task.id),
    startRename,
    onCloseAutoFocus,
    statuses,
    pickStatus,
    askArchive: () => setIsConfirmingArchive(true),
    keep: () => setIsConfirmingArchive(false),
    confirmArchive,
    isConfirmingArchive,
    isPending: actions.isPending,
    run: RUN_ACTIONS[runAction],
    pickRun,
    closeRunDialog,
    isConfirmingStop: runDialog === "stop",
    isAskingRetry: runDialog === "retry",
    // A repeating task's Stop ends only the run going now.
    confirmStop: () =>
      task.repeat
        ? actions.stopRun(closeRunDialog)
        : actions.stop(closeRunDialog),
    stopRepeating: actions.stopRepeating,
    retry: (instructions: string) => actions.run(instructions, closeRunDialog),
    isStarting: actions.isStarting,
    isStopping: task.repeat ? actions.isStoppingRun : actions.isStopping,
  };
}

export type TaskMenuState = ReturnType<typeof useTaskMenu>;
