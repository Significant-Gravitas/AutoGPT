"use client";

import { RenameField } from "@/app/(platform)/tasks/components/TaskMenu/components/RenameField/RenameField";
import { TaskMenu } from "@/app/(platform)/tasks/components/TaskMenu/TaskMenu";
import type { Task } from "@/app/api/__generated__/models/task";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";
import { taskTabKey, useWorkAreaStore } from "../../workArea";
import { CopyLinkButton } from "./CopyLinkButton";
import { TaskControls } from "./TaskControls/TaskControls";

interface Props {
  task: Task | null;
  /** The task's name while the task itself is still loading. */
  title: string | null;
}

/** The top of a task's tab: its whole name with the same ⋯ menu as its
 *  card (run or stop, rename, change the status, archive), then the
 *  buttons for what it is doing now. Archiving closes the tab. */
export function TaskHeader({ task, title }: Props) {
  const [isRenaming, setIsRenaming] = useState(false);
  const close = useWorkAreaStore((s) => s.close);
  const name = task?.title ?? title;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex min-h-7 items-start gap-2">
        {task && isRenaming ? (
          <RenameField
            taskID={task.id}
            title={task.title}
            onDone={() => setIsRenaming(false)}
          />
        ) : name ? (
          <Text
            variant="body-medium"
            as="h2"
            dir="auto"
            className="min-w-0 flex-1 text-pretty break-words"
          >
            {name}
          </Text>
        ) : (
          <Skeleton className="mt-1 h-4 w-2/3" />
        )}
        {task ? <CopyLinkButton task={task} /> : null}
        {task ? (
          <TaskMenu
            task={task}
            onRename={() => setIsRenaming(true)}
            onArchived={() => close(taskTabKey(task.id))}
            className="shrink-0"
          />
        ) : null}
      </div>
      {task ? <TaskControls task={task} /> : null}
    </div>
  );
}
