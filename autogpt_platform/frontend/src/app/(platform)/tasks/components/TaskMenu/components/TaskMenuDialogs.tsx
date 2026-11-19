"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import type { RefObject } from "react";
import { StopTaskDialog } from "../../StopTaskDialog/StopTaskDialog";
import { TryAgainDialog } from "../../TryAgainForm/TryAgainDialog";
import type { TaskMenuState } from "../useTaskMenu";
import { ArchiveTaskDialog } from "./ArchiveTaskDialog/ArchiveTaskDialog";
import { DuplicateTaskDialog } from "./DuplicateTaskDialog";

interface Props {
  task: Task;
  menu: TaskMenuState;
  triggerRef: RefObject<HTMLElement | null>;
  onDuplicated?: (id: string) => void;
}

/** What the menu asks before it acts: archive, stop, try again with a
 *  note, and the create dialog for a copy. */
export function TaskMenuDialogs({
  task,
  menu,
  triggerRef,
  onDuplicated,
}: Props) {
  return (
    <>
      <ArchiveTaskDialog
        isOpen={menu.isConfirmingArchive}
        title={task.title}
        isPending={menu.isPending}
        onCancel={menu.keep}
        onConfirm={menu.confirmArchive}
      />
      <StopTaskDialog
        isOpen={menu.isConfirmingStop}
        title={task.title}
        isPending={menu.isStopping}
        onCancel={menu.closeRunDialog}
        onConfirm={menu.confirmStop}
        isRepeating={!!task.repeat}
      />
      <TryAgainDialog
        isOpen={menu.isAskingRetry}
        taskID={task.id}
        isPending={menu.isStarting}
        onCancel={menu.closeRunDialog}
        onSubmit={menu.retry}
      />
      <DuplicateTaskDialog
        task={task}
        isOpen={menu.isDuplicating}
        onOpenChange={menu.setDuplicating}
        onCreated={onDuplicated}
        returnFocusRef={triggerRef}
      />
    </>
  );
}
