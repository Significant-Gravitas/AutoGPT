"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { taskRowID, type TaskOwner } from "../../../../helpers";
import { NowLine } from "../../../NowLine/NowLine";
import { OwnerChip } from "../../../OwnerChip/OwnerChip";
import { StateIcon } from "../../../StateIcon/StateIcon";
import { StatusPill } from "../../../StatusPill/StatusPill";
import { useListMotion } from "../../../TaskList/useListMotion";
import { RenameField } from "../../../TaskMenu/components/RenameField/RenameField";
import { TaskMenu } from "../../../TaskMenu/TaskMenu";
import { useTaskRow } from "../../../TaskRow/useTaskRow";
import type { CardNav } from "../../useRovingCards";

interface Props {
  task: Task;
  /** Who works it; null while Otto is the only one who can. */
  owner: TaskOwner | null;
  selected: boolean;
  onOpen: () => void;
  nav: CardNav;
}

/** One task on the board: its title, what it is doing now, who works it,
 *  its status and age. */
export function BoardCard({ task, owner, selected, onOpen, nav }: Props) {
  const motionProps = useListMotion(`board-task-${task.id}`);
  const card = useTaskRow(task);

  return (
    <motion.li
      {...motionProps}
      data-testid="board-card"
      data-status={task.status}
      data-schedule={card.schedule ?? undefined}
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
          data-task-card={task.id}
          tabIndex={nav.isTabStop ? 0 : -1}
          aria-current={selected ? "true" : undefined}
          onClick={onOpen}
          onFocus={nav.onFocus}
          onKeyDown={nav.onKeyDown}
          className="grid w-full gap-1.5 rounded-card py-3 pr-3 pl-3 text-left focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
        >
          <span className="flex min-w-0 items-start gap-2 pr-7">
            <StateIcon
              status={task.status}
              schedule={card.schedule}
              queued={card.queued}
              className="mt-0.5"
            />
            <Text
              as="span"
              variant="inbox-medium"
              tone="ink"
              className="line-clamp-2 min-w-0 break-words"
              dir="auto"
            >
              {task.title}
            </Text>
          </span>
          <NowLine task={task} className="pl-6" />
          {owner ? (
            <span className="flex min-w-0 pl-6">
              <OwnerChip owner={owner} />
            </span>
          ) : null}
          <span className="flex min-w-0 items-center justify-between gap-2 pl-6">
            <StatusPill
              status={task.status}
              schedule={card.schedule}
              queued={card.queued}
            />
            {card.age ? (
              <Text
                as="span"
                variant="inbox"
                tone="faint"
                tabular
                className="min-w-0 truncate"
                title={card.age}
              >
                {card.age}
              </Text>
            ) : null}
          </span>
          <span className="sr-only">{nav.position}</span>
        </button>
      )}
      <div className="absolute top-2 right-2">
        <TaskMenu
          task={task}
          onRename={card.startRename}
          tabIndex={nav.isTabStop ? 0 : -1}
        />
      </div>
    </motion.li>
  );
}
