import type { Expert } from "@/app/api/__generated__/models/expert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type { RefObject } from "react";
import { NewTaskForm } from "./components/NewTaskForm";
import type { TaskDraft } from "./draft";
import { useReturnFocus } from "./useReturnFocus";

interface Props {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  experts: Expert[];
  onCreated?: (id: string) => void;
  /** A task to start from (Duplicate, a suggestion), instead of the
   *  kept draft. */
  initial?: TaskDraft;
  /** What gets focus back when the dialog closes: the button that
   *  opened it. */
  returnFocusRef?: RefObject<HTMLElement | null>;
}

/** Write a task, pick who takes it, and, under Options, when it starts or
 *  how often it runs and what done looks like. */
export function NewTaskDialog({
  isOpen,
  onOpenChange,
  experts,
  onCreated,
  initial,
  returnFocusRef,
}: Props) {
  useReturnFocus(isOpen, returnFocusRef);

  return (
    <Dialog
      title="New task"
      className="max-w-xl max-lg:inset-x-0 max-lg:mx-auto"
      controlled={{ isOpen, set: onOpenChange }}
    >
      <Dialog.Content>
        <NewTaskForm
          experts={experts}
          onOpenChange={onOpenChange}
          onCreated={onCreated}
          initial={initial}
        />
      </Dialog.Content>
    </Dialog>
  );
}
