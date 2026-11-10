import type { Task } from "@/app/api/__generated__/models/task";
import { useRef, useState } from "react";
import { STATUS_ORDER } from "../../helpers";
import { useTaskActions } from "../../useTaskActions";

interface Args {
  task: Task;
  onRename: () => void;
  onArchived?: () => void;
}

export function useTaskMenu({ task, onRename, onArchived }: Args) {
  const actions = useTaskActions(task.id);
  const [isConfirmingArchive, setIsConfirmingArchive] = useState(false);
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
    const status = STATUS_ORDER.find((s) => s === value);
    if (!status || status === task.status) return;
    actions.setStatus(status);
  }

  function confirmArchive() {
    actions.archiveTask(() => {
      setIsConfirmingArchive(false);
      onArchived?.();
    });
  }

  return {
    startRename,
    onCloseAutoFocus,
    pickStatus,
    askArchive: () => setIsConfirmingArchive(true),
    keep: () => setIsConfirmingArchive(false),
    confirmArchive,
    isConfirmingArchive,
    isPending: actions.isPending,
  };
}
