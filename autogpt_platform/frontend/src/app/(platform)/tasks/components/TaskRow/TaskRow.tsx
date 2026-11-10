"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { taskRowID, type TaskOwner } from "../../helpers";
import { OwnerChip } from "../OwnerChip/OwnerChip";
import { StateIcon } from "../StateIcon/StateIcon";
import { StatusPill } from "../StatusPill/StatusPill";
import { RenameField } from "../TaskMenu/components/RenameField/RenameField";
import { TaskMenu } from "../TaskMenu/TaskMenu";
import { useTaskRow } from "./useTaskRow";

interface Props {
  task: Task;
  owner: TaskOwner;
  selected: boolean;
  onOpen: () => void;
}

/** One task in the list: its mark, title and status on the left, who works
 *  it and for how long on the right. */
export function TaskRow({ task, owner, selected, onOpen }: Props) {
  const row = useTaskRow(task);

  return (
    <div
      data-testid="task-row"
      data-status={task.status}
      data-selected={selected || undefined}
      className={cn(
        "flex min-w-0 items-center transition-colors duration-150 ease-out-expo hover:bg-surface-wash motion-reduce:transition-none",
        selected && "bg-surface-wash",
      )}
    >
      {row.isRenaming ? (
        <div className="min-w-0 flex-1 py-2 pl-2">
          <RenameField
            taskID={task.id}
            title={task.title}
            onDone={row.stopRename}
          />
        </div>
      ) : (
        <button
          type="button"
          id={taskRowID(task.id)}
          aria-current={selected ? "true" : undefined}
          onClick={onOpen}
          className="flex min-w-0 flex-1 items-center gap-3 py-3 pr-2 pl-4 text-left transition-transform duration-150 ease-out-expo focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none focus-visible:ring-inset active:scale-99 motion-reduce:transition-none"
        >
          <StateIcon status={task.status} />
          <span className="flex min-w-0 flex-1 items-center gap-2">
            <Text
              as="span"
              variant="inbox-medium"
              tone="ink"
              className="min-w-0 truncate"
            >
              {task.title}
            </Text>
            <StatusPill status={task.status} />
          </span>
          <span className="hidden max-w-40 shrink-0 md:inline-flex">
            <OwnerChip owner={owner} />
          </span>
          {row.age ? (
            <Text
              as="span"
              variant="inbox"
              tone="faint"
              tabular
              className="shrink-0 whitespace-nowrap"
            >
              {row.age}
            </Text>
          ) : null}
        </button>
      )}
      <TaskMenu task={task} onRename={row.startRename} className="mr-2" />
    </div>
  );
}
