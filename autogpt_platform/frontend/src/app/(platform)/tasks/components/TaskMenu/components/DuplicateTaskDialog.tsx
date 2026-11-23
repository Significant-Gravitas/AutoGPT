"use client";

import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import type { Task } from "@/app/api/__generated__/models/task";
import { okData } from "@/app/api/helpers";
import type { RefObject } from "react";
import { draftFromTask } from "../../NewTaskDialog/draft";
import { NewTaskDialog } from "../../NewTaskDialog/NewTaskDialog";

interface Props {
  task: Task;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated?: (id: string) => void;
  returnFocusRef: RefObject<HTMLElement | null>;
}

/** The create dialog filled in from a task: its words, goal, finish line
 *  and owner, not its schedule, ready to change before it is sent. */
export function DuplicateTaskDialog({
  task,
  isOpen,
  onOpenChange,
  onCreated,
  returnFocusRef,
}: Props) {
  const experts = useListExperts({
    query: { select: (res) => okData(res) ?? [], enabled: isOpen },
  });

  return (
    <NewTaskDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      experts={experts.data ?? []}
      initial={draftFromTask(task)}
      onCreated={onCreated}
      returnFocusRef={returnFocusRef}
    />
  );
}
