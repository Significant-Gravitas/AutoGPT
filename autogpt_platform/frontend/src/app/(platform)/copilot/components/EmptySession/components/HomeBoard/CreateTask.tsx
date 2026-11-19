"use client";

import { NewTaskDialog } from "@/app/(platform)/tasks/components/NewTaskDialog/NewTaskDialog";
import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { TaskAdd01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { useWorkAreaStore } from "../../../../workArea";

/** A tab's name until its task is read. */
const UNREAD_TITLE = "Task";

/** The board's one coloured button: write a task, pick who takes it, and it
 *  opens beside the board. */
export function CreateTask() {
  const [isOpen, setIsOpen] = useState(false);
  const openTask = useWorkAreaStore((s) => s.openTask);
  const expertsQuery = useListExperts({
    query: { select: (res) => okData(res) ?? [] },
  });

  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-gradient-to-b from-info to-info-foreground pr-3 pl-2.5 text-body-sm font-medium whitespace-nowrap text-surface shadow-(--inbox-shadow-lit) transition-[transform,filter] duration-150 ease-out outline-none hover:brightness-110 focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:ring-offset-2 active:scale-[0.97] motion-reduce:transition-none"
      >
        <Icon icon={TaskAdd01Icon} size={15} tone="inherit" />
        Create task
      </button>
      <NewTaskDialog
        isOpen={isOpen}
        onOpenChange={setIsOpen}
        experts={expertsQuery.data ?? []}
        onCreated={(id) => openTask(id, UNREAD_TITLE)}
      />
    </>
  );
}
