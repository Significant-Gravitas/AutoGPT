"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { taskKey } from "../../data/helpers";
import type { Task } from "../../data/types";
import { isCancellable } from "../../helpers";
import { CancelTask } from "../CancelTask/CancelTask";
import { CardApprovals } from "../CardApprovals/CardApprovals";
import { StateDot } from "../StateDot/StateDot";
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
  const ask = cards.length ? null : task.ask;

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
      <button
        type="button"
        id={`inbox-row-${task.id}`}
        aria-current={selected ? "true" : undefined}
        onClick={onOpen}
        className={cn(
          "flex min-w-0 flex-col gap-1 text-left transition-transform duration-150 ease-out-expo focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none focus-visible:ring-inset active:scale-99 motion-reduce:transition-none",
          board ? "rounded-card p-3" : "py-3 pr-4 pl-3",
        )}
      >
        <span className="flex min-w-0 items-start gap-2">
          <StateDot state={task.state} className="mt-0.5" />
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
        <span className="flex min-w-0 items-baseline gap-1.5 pl-6">
          <Text
            as="span"
            variant="inbox"
            tone={task.state === "stuck" ? "warning" : "quiet"}
            className="line-clamp-2 min-w-0"
          >
            {task.now || "Not started"}
          </Text>
          <Text
            as="span"
            variant="inbox"
            tone="faint"
            tabular
            className="shrink-0"
          >
            · {row.ago}
          </Text>
        </span>
        {board ? (
          <span className="flex min-w-0 items-center gap-2 pt-1 pl-6">
            <span className="flex min-w-0 flex-1 overflow-hidden">
              <RowAssignee assignee={task.assignee} />
            </span>
            <Text as="span" variant="small" tone="faint" mono>
              {taskKey(task.number)}
            </Text>
          </span>
        ) : null}
      </button>
      {ask || row.action || cards.length ? (
        <div
          className={cn(
            "flex flex-col gap-2 rounded-control bg-surface-wash p-2.5",
            board ? "mx-3 mb-3" : "mr-4 mb-3 ml-9",
          )}
        >
          {ask ? (
            <Text variant="inbox" tone="ink" className="break-words">
              {ask.question}
            </Text>
          ) : null}
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
        <div className={cn("flex", board ? "px-3 pb-2.5" : "mr-4 mb-2.5 ml-9")}>
          <CancelTask taskId={task.id} />
        </div>
      ) : null}
    </div>
  );
}
