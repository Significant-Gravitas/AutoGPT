"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { taskRowID, type TaskOwner } from "../../../../helpers";
import { OwnerChip } from "../../../OwnerChip/OwnerChip";
import { StateIcon } from "../../../StateIcon/StateIcon";
import { StatusPill } from "../../../StatusPill/StatusPill";
import { RenameField } from "../../../TaskMenu/components/RenameField/RenameField";
import { TaskMenu } from "../../../TaskMenu/TaskMenu";
import { useTaskRow } from "../../../TaskRow/useTaskRow";
import { useListMotion } from "../../useListMotion";

interface Props {
  task: Task;
  owner: TaskOwner;
  selected: boolean;
  onOpen: () => void;
}

/** One task in the list, on a single line: its state, title, who works
 *  it, its status and age, like a board card laid flat. */
export function ListRow({ task, owner, selected, onOpen }: Props) {
  const motionProps = useListMotion(`list-task-${task.id}`);
  const row = useTaskRow(task);

  return (
    <motion.li
      {...motionProps}
      data-testid="list-row"
      data-title={task.title}
      data-status={task.status}
      data-schedule={row.schedule ?? undefined}
      data-selected={selected || undefined}
      className={cn(
        "relative min-w-0 rounded-card bg-surface shadow-(--inbox-shadow-card) transition-shadow duration-200 hover:shadow-(--inbox-shadow-card-hover) motion-reduce:transition-none",
        selected && "shadow-(--inbox-shadow-card-hover)",
      )}
    >
      {row.isRenaming ? (
        <div className="py-1.5 pr-11 pl-2">
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
          className="flex w-full min-w-0 items-center gap-3 rounded-card py-2.5 pr-11 pl-3 text-left focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
        >
          <StateIcon status={task.status} schedule={row.schedule} />
          <Text
            as="span"
            variant="inbox-medium"
            tone="ink"
            className="min-w-0 flex-1 truncate"
            title={task.title}
          >
            {task.title}
          </Text>
          <span className="hidden max-w-40 min-w-0 shrink-0 md:flex">
            <OwnerChip owner={owner} />
          </span>
          <span className="flex shrink-0">
            <StatusPill status={task.status} schedule={row.schedule} />
          </span>
          {row.age ? (
            <Text
              as="span"
              variant="inbox"
              tone="faint"
              tabular
              className="hidden max-w-40 min-w-0 shrink-0 truncate text-right sm:block"
              title={row.age}
            >
              {row.age}
            </Text>
          ) : null}
        </button>
      )}
      <div className="absolute top-1/2 right-2 -translate-y-1/2">
        <TaskMenu task={task} onRename={row.startRename} />
      </div>
    </motion.li>
  );
}
