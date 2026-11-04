"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { taskKey } from "../../data/helpers";
import type { Task } from "../../data/types";
import { isCancellable } from "../../helpers";
import { CancelTask } from "../CancelTask/CancelTask";
import { CardApprovals } from "../CardApprovals/CardApprovals";
import { StatusBadge } from "../StatusBadge/StatusBadge";
import { RenameField } from "../TaskMenu/components/RenameField/RenameField";
import { TaskMenu } from "../TaskMenu/TaskMenu";
import { RowAssignee } from "./components/RowAssignee";
import { openLabel } from "./helpers";
import { useInboxRow } from "./useInboxRow";

interface Props {
  task: Task;
  onOpen(): void;
  variant?: "list" | "board";
  selected?: boolean;
}

export function InboxRow({
  task,
  onOpen,
  variant = "list",
  selected = false,
}: Props) {
  const row = useInboxRow(task);
  const board = variant === "board";
  const cards = row.cards;

  return (
    <div
      data-testid="inbox-row"
      data-state={task.state}
      data-selected={selected || undefined}
      className={cn(
        "flex flex-col transition-colors duration-150 ease-out-expo motion-reduce:transition-none",
        board
          ? "rounded-card bg-surface shadow-card transition-shadow hover:shadow-card-hover"
          : "hover:bg-surface-wash",
        board && selected && "shadow-card-hover",
        !board && selected && "bg-surface-wash",
      )}
    >
      <div className="flex min-w-0 items-start">
        {row.isRenaming ? (
          <div className={cn("min-w-0 flex-1", board ? "p-2" : "py-2 pl-2")}>
            <RenameField
              taskId={task.id}
              title={task.title}
              onDone={row.stopRename}
            />
          </div>
        ) : (
          <button
            type="button"
            id={`inbox-row-${task.id}`}
            aria-current={selected ? "true" : undefined}
            onClick={onOpen}
            className={cn(
              "flex min-w-0 flex-1 flex-col gap-1 text-left transition-transform duration-150 ease-out-expo focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none focus-visible:ring-inset active:scale-99 motion-reduce:transition-none",
              board ? "rounded-card p-3" : "py-3 pr-2 pl-4",
            )}
          >
            <span className="flex min-w-0 items-start gap-2">
              <Text
                as="span"
                variant="inbox-medium"
                tone="ink"
                className={cn(
                  "min-w-0 flex-1 break-words",
                  board ? "line-clamp-2" : "truncate",
                )}
              >
                {task.title}
              </Text>
              {board ? null : (
                <span className="hidden shrink-0 md:inline-flex">
                  <RowAssignee assignee={task.assignee} />
                </span>
              )}
            </span>
            <span className="flex min-w-0 items-baseline gap-1.5">
              <StatusBadge state={task.state} className="self-center" />
              {row.line ? (
                <Text
                  as="span"
                  variant="inbox"
                  tone="quiet"
                  className="min-w-0 truncate"
                  title={row.line}
                >
                  {row.line}
                </Text>
              ) : null}
              {row.age ? (
                <Text
                  as="span"
                  variant="inbox"
                  tone="faint"
                  tabular
                  className="shrink-0 whitespace-nowrap"
                >
                  · {row.age}
                </Text>
              ) : null}
            </span>
            {board ? (
              <span className="flex min-w-0 items-center gap-2 pt-1">
                <span className="flex min-w-0 flex-1 overflow-hidden">
                  <RowAssignee assignee={task.assignee} />
                </span>
                <Text as="span" variant="small" tone="faint" mono>
                  {taskKey(task.number)}
                </Text>
              </span>
            ) : null}
          </button>
        )}
        <TaskMenu
          taskId={task.id}
          title={task.title}
          onRename={row.startRename}
          className="mt-2 mr-2 shrink-0"
        />
      </div>
      {row.action || cards.length ? (
        <div
          className={cn(
            "flex flex-col gap-2 rounded-control bg-surface-wash p-2.5",
            board ? "mx-3 mb-3" : "mr-4 mb-3 ml-4",
          )}
        >
          {cards.length ? (
            <CardApprovals taskId={task.id} cards={cards} />
          ) : (
            <div className="flex flex-wrap items-center gap-1.5">
              {row.action ? (
                <Button
                  variant="primary"
                  size="xs"
                  className="max-w-full"
                  title={row.action.label}
                  loading={row.busy}
                  onClick={row.runAction}
                >
                  <span className="truncate">{row.action.label}</span>
                </Button>
              ) : null}
              <Button variant="ghost-muted" size="xs" onClick={onOpen}>
                {openLabel(task, !!row.action)}
              </Button>
            </div>
          )}
        </div>
      ) : null}
      {isCancellable(task.state) ? (
        <div className={cn("flex", board ? "px-3 pb-2.5" : "mr-4 mb-2.5 ml-4")}>
          <CancelTask taskId={task.id} />
        </div>
      ) : null}
    </div>
  );
}
