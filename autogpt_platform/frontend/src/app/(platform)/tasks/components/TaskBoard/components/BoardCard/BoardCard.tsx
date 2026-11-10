"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { taskRowID, type TaskOwner } from "../../../../helpers";
import { OwnerChip } from "../../../OwnerChip/OwnerChip";
import { StateIcon } from "../../../StateIcon/StateIcon";
import { StatusPill } from "../../../StatusPill/StatusPill";
import { useListMotion } from "../../../TaskList/useListMotion";
import { RenameField } from "../../../TaskMenu/components/RenameField/RenameField";
import { TaskMenu } from "../../../TaskMenu/TaskMenu";
import { useTaskRow } from "../../../TaskRow/useTaskRow";

interface Props {
  task: Task;
  owner: TaskOwner;
  selected: boolean;
  onOpen: () => void;
}

/** One task on the board: its title, who works it, its status and age. */
export function BoardCard({ task, owner, selected, onOpen }: Props) {
  const motionProps = useListMotion(`board-task-${task.id}`);
  const card = useTaskRow(task);

  return (
    <motion.li
      {...motionProps}
      data-testid="board-card"
      data-status={task.status}
      data-selected={selected || undefined}
      className={cn(
        "relative rounded-card bg-surface shadow-(--inbox-shadow-card) transition-shadow duration-200 hover:shadow-(--inbox-shadow-card-hover) motion-reduce:transition-none",
        selected && "shadow-(--inbox-shadow-card-hover)",
      )}
    >
      {card.isRenaming ? (
        <div className="py-2 pr-11 pl-2">
          <RenameField
            taskID={task.id}
            title={task.title}
            onDone={card.stopRename}
          />
        </div>
      ) : (
        <button
          type="button"
          id={taskRowID(task.id)}
          aria-current={selected ? "true" : undefined}
          onClick={onOpen}
          className="grid w-full gap-1.5 rounded-card py-3 pr-3 pl-3 text-left focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
        >
          <span className="flex min-w-0 items-start gap-2 pr-7">
            <StateIcon status={task.status} className="mt-0.5" />
            <Text
              as="span"
              variant="inbox-medium"
              tone="ink"
              className="line-clamp-2 min-w-0 break-words"
            >
              {task.title}
            </Text>
          </span>
          <span className="flex min-w-0 pl-6">
            <OwnerChip owner={owner} />
          </span>
          <span className="flex min-w-0 items-center justify-between gap-2 pl-6">
            <StatusPill status={task.status} />
            {card.age ? (
              <Text
                as="span"
                variant="inbox"
                tone="faint"
                tabular
                className="whitespace-nowrap"
              >
                {card.age}
              </Text>
            ) : null}
          </span>
        </button>
      )}
      <div className="absolute top-2 right-2">
        <TaskMenu task={task} onRename={card.startRename} />
      </div>
    </motion.li>
  );
}
