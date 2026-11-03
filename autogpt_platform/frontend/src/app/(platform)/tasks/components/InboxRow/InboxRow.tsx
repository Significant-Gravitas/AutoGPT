"use client";

import { cn } from "@/lib/utils";
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
        board && "rounded-xl border border-border-card bg-surface shadow-sm",
        board && selected && "border-border-strong",
      )}
    >
      <button
        type="button"
        id={`inbox-row-${task.id}`}
        aria-expanded={expanded}
        aria-controls={detailsID}
        aria-current={selected ? "true" : undefined}
        onClick={onToggle}
        className={cn(
          "w-full text-left transition-[background-color,transform] duration-150 ease-out-expo hover:bg-surface-muted focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none focus-visible:ring-inset active:scale-99 motion-reduce:transition-none",
          board ? "rounded-xl p-3" : "px-4 py-3",
          expanded && !board && "bg-surface-muted",
          selected && "bg-surface-tray/60 hover:bg-surface-tray/60",
        )}
      >
        <RowSummary task={task} elapsed={row.elapsed} variant={variant} />
      </button>
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
