"use client";

import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { tourTarget } from "@/lib/tour-targets";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, ArrowUp01Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence } from "framer-motion";
import { useId } from "react";
import type { ExpertIdentity, ExpertIdentityMap } from "../../useExpertMap";
import { TaskRow } from "./components/TaskRow";
import {
  focusTaskRow,
  getNeighbourID,
  getStartHereID,
  getTaskOwner,
  splitTasks,
} from "./helpers";

interface Props {
  /** Every task in stored (ranked) order, removed ones included. */
  tasks: SuggestedTask[];
  experts: ExpertIdentity[];
  /** Fired experts too, for the rows already started. */
  expertsById: ExpertIdentityMap;
  sendingId: string | null;
  disabled?: boolean;
  /** Whether the user opened or closed "more tasks"; null until they do.
   *  Kept above the collapse, so Hide and Show do not reset it. */
  moreChoice: boolean | null;
  onMoreChoice: (isOpen: boolean) => void;
  /** Where focus goes when a removed row has no neighbour left. */
  fallbackFocusID: string;
  onSend: (task: SuggestedTask) => void;
  onRemove: (task: SuggestedTask) => void;
  className?: string;
}

// The ranked tasks, each with who takes it. The first three are the plan;
// the walkthrough's spotlight sits on their list, and the rest wait under
// "{n} more tasks" until one of the three is started or removed.
export function SuggestedTaskList({
  tasks,
  experts,
  expertsById,
  sendingId,
  disabled = false,
  moreChoice,
  onMoreChoice,
  fallbackFocusID,
  onSend,
  onRemove,
  className,
}: Props) {
  const { top, more, hasActedOnTop } = splitTasks(tasks);
  const isMoreOpen = moreChoice ?? hasActedOnTop;
  const moreID = useId();
  const startHereID = getStartHereID([...top, ...(isMoreOpen ? more : [])]);
  const sendingTask = tasks.find((task) => task.id === sendingId);

  function handleRemove(task: SuggestedTask) {
    const isTop = top.some((row) => row.id === task.id);
    // Acting on one of the three brings up the rest, and keeps them up
    // through an Undo.
    const willMoreOpen = isMoreOpen || (isTop && moreChoice === null);
    if (isTop && moreChoice === null) onMoreChoice(true);
    const rows = [...top, ...(willMoreOpen ? more : [])];
    // The last task takes the block, and its toggle, with it.
    const isLast = top.length + more.length === 1;
    onRemove(task);
    focusTaskRow(
      getNeighbourID(rows, task),
      isLast ? undefined : fallbackFocusID,
    );
  }

  function renderRows(rows: SuggestedTask[], showWhy: boolean) {
    return (
      <AnimatePresence initial={false}>
        {rows.map((task) => (
          <TaskRow
            key={task.id}
            task={task}
            owner={getTaskOwner(task, experts, expertsById)}
            isStartHere={task.id === startHereID}
            showWhy={showWhy}
            isBusy={disabled || sendingId !== null}
            isSending={sendingId === task.id}
            isRemoveDisabled={sendingId !== null}
            onSend={onSend}
            onRemove={handleRemove}
          />
        ))}
      </AnimatePresence>
    );
  }

  return (
    <div
      className={cn(
        "overflow-hidden rounded-lg border border-border-base bg-surface",
        className,
      )}
    >
      <ol
        {...tourTarget("first-tasks")}
        aria-label="Your first tasks"
        className="divide-y divide-separator"
      >
        {renderRows(top, true)}
      </ol>
      {more.length > 0 ? (
        <div className={cn(top.length > 0 && "border-t border-separator")}>
          <button
            type="button"
            aria-expanded={isMoreOpen}
            aria-controls={moreID}
            onClick={() => onMoreChoice(!isMoreOpen)}
            className="flex min-h-[3.25rem] w-full items-center gap-3 px-4 py-3 text-left outline-none transition-colors duration-150 hover:bg-surface-muted focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus"
          >
            <Text variant="body-medium" as="span" tone="secondary">
              {more.length} more {more.length === 1 ? "task" : "tasks"}
            </Text>
            <Icon
              icon={isMoreOpen ? ArrowUp01Icon : ArrowDown01Icon}
              size={15}
              className="ml-auto shrink-0 text-foreground-tertiary"
            />
          </button>
          <ol
            id={moreID}
            hidden={!isMoreOpen}
            aria-label="More tasks"
            className="divide-y divide-separator border-t border-separator"
          >
            {isMoreOpen ? renderRows(more, false) : null}
          </ol>
        </div>
      ) : null}
      <p role="status" className="sr-only">
        {sendingTask ? `Starting: ${sendingTask.title}` : ""}
      </p>
    </div>
  );
}
