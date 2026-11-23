import type { TaskDraft } from "@/app/(platform)/tasks/components/NewTaskDialog/draft";
import { useRef, useState } from "react";

/** The Tasks view's one create dialog, opened from Create task, the empty
 *  board, a suggestion or the N key; focus goes back to Create task. */
export function useTaskLauncher() {
  const [isOpen, setIsOpen] = useState(false);
  const [initial, setInitial] = useState<TaskDraft | undefined>(undefined);
  const openerRef = useRef<HTMLButtonElement>(null);

  return {
    isOpen,
    initial,
    openerRef,
    /** Open it, empty or kept draft, or starting from `from`. */
    open(from?: TaskDraft) {
      setInitial(from);
      setIsOpen(true);
    },
    setOpen: setIsOpen,
  };
}

export type TaskLauncher = ReturnType<typeof useTaskLauncher>;
