import { useRef, useState } from "react";
import { useDeleteTask } from "../../data/useTasksData";

interface Args {
  taskId: string;
  onRename: () => void;
  onDeleted?: () => void;
}

export function useTaskMenu({ taskId, onRename, onDeleted }: Args) {
  const deleteTask = useDeleteTask();
  const [isConfirmingDelete, setIsConfirmingDelete] = useState(false);
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

  function confirmDelete() {
    setIsConfirmingDelete(false);
    deleteTask.mutate(taskId);
    onDeleted?.();
  }

  return {
    startRename,
    onCloseAutoFocus,
    askDelete: () => setIsConfirmingDelete(true),
    keep: () => setIsConfirmingDelete(false),
    confirmDelete,
    isConfirmingDelete,
  };
}
