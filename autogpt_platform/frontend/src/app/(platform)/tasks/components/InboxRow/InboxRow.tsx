"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence } from "framer-motion";
import type { Task } from "../../data/types";
import { FoldPanel } from "./components/FoldPanel";
import { RowDetails } from "./components/RowDetails";
import { RowSummary } from "./components/RowSummary";
import { useInboxRow } from "./useInboxRow";

interface Props {
  task: Task;
  expanded: boolean;
  onToggle(): void;
  onOpen(): void;
  variant?: "list" | "board";
  selected?: boolean;
}

export function InboxRow({
  task,
  expanded,
  onToggle,
  onOpen,
  variant = "list",
  selected = false,
}: Props) {
  const row = useInboxRow(task, expanded);
  const board = variant === "board";
  const detailsID = `inbox-row-${task.id}-details`;

  return (
    <div
      data-testid="inbox-row"
      data-state={task.inbox.state}
      data-selected={selected || undefined}
      onKeyDown={row.handleKeyDown}
      className={cn(
        "flex flex-col",
        board &&
          "rounded-card bg-surface shadow-card transition-shadow duration-150 ease-out-expo hover:shadow-card-hover motion-reduce:transition-none",
        board && selected && "shadow-card-hover",
      )}
    >
      <div
        className={cn(
          "group/row flex items-stretch transition-colors duration-150 ease-out-expo motion-reduce:transition-none",
          board ? "rounded-card" : "hover:bg-surface-wash",
          selected && "bg-surface-wash",
        )}
      >
        <button
          type="button"
          id={`inbox-row-${task.id}`}
          aria-current={selected ? "true" : undefined}
          onClick={onOpen}
          className={cn(
            "min-w-0 flex-1 text-left transition-transform duration-150 ease-out-expo focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none focus-visible:ring-inset active:scale-99 motion-reduce:transition-none",
            board ? "rounded-l-card p-3 pr-1" : "py-3 pr-1 pl-4",
          )}
        >
          <RowSummary task={task} elapsed={row.elapsed} variant={variant} />
        </button>
        <button
          type="button"
          aria-expanded={expanded}
          aria-controls={detailsID}
          aria-label={expanded ? "Hide steps" : "Show steps"}
          onClick={onToggle}
          className={cn(
            "flex w-8 shrink-0 items-start justify-center pt-3.5 focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none focus-visible:ring-inset",
            board ? "rounded-r-card" : "mr-1 rounded-control",
          )}
        >
          <span
            aria-hidden
            className={cn(
              "flex transition-transform duration-300 ease-out-expo motion-reduce:transition-none",
              expanded && "rotate-180",
            )}
          >
            <Icon
              icon={ArrowDown01Icon}
              size={16}
              tone={expanded ? "ink" : "faint"}
            />
          </span>
        </button>
      </div>
      <AnimatePresence initial={false}>
        {expanded ? (
          <FoldPanel
            key="details"
            id={detailsID}
            labelledBy={`inbox-row-${task.id}`}
          >
            <div className={cn(board ? "px-3 pb-3" : "pt-1 pr-4 pb-4 pl-11")}>
              <RowDetails
                task={task}
                now={row.now}
                action={row.action}
                busy={row.busy}
                pending={row.pending}
                onAnswer={row.answerWith}
                onApprove={row.approve}
                onReject={row.reject}
                onRetry={row.retry}
                onOpen={onOpen}
              />
            </div>
          </FoldPanel>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
