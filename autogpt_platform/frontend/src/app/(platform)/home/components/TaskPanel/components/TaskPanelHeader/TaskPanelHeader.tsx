"use client";

import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { SheetTitle } from "@/components/ui/sheet";
import {
  ArrowExpand01Icon,
  CheckmarkCircle02Icon,
  Folder01Icon,
  UserSwitchIcon,
} from "@hugeicons/core-free-icons";
import type { Card } from "../../../../board-helpers";
import { AssigneePicker } from "../../../AssigneePicker/AssigneePicker";
import { useTaskPanelHeader } from "./useTaskPanelHeader";

interface Props {
  sessionId: string;
  title: string;
  card: Card | null;
  columns: BoardColumn[];
  columnKey: string | null;
  showFiles: boolean;
  onToggleFiles: () => void;
  onRename: (title: string) => void;
  onAssign: (expertId: string | null) => void;
  onMove: (column: string) => void;
  onReassign: (card: Card) => void;
}

export function TaskPanelHeader({
  sessionId,
  title,
  card,
  columns,
  columnKey,
  showFiles,
  onToggleFiles,
  onRename,
  onAssign,
  onMove,
  onReassign,
}: Props) {
  const {
    isEditing,
    canReassign,
    draft,
    startEditing,
    stopEditing,
    handleChange,
    handleKeyDown,
  } = useTaskPanelHeader({ title, card, onRename });

  function handleMarkDone() {
    onMove("done");
  }

  return (
    <header className="flex shrink-0 flex-col gap-3 border-b border-zinc-200 bg-white px-5 pb-3 pt-4">
      <div className="flex min-h-8 items-center pr-8">
        {isEditing ? (
          <>
            <SheetTitle className="sr-only">{title}</SheetTitle>
            <Input
              id={`task-title-${sessionId}`}
              label="Task title"
              hideLabel
              size="small"
              autoFocus
              autoComplete="off"
              value={draft}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              onBlur={stopEditing}
              wrapperClassName="mb-0 w-full"
            />
          </>
        ) : (
          <SheetTitle className="min-w-0 text-base font-semibold text-zinc-900">
            <button
              type="button"
              onClick={startEditing}
              title="Rename"
              className="max-w-full truncate rounded-md px-1 py-0.5 text-left hover:bg-zinc-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800"
            >
              {title}
            </button>
          </SheetTitle>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {card && !card.started && (
          <AssigneePicker
            expertId={card.expert_id ?? null}
            onSelect={onAssign}
          />
        )}
        {card && canReassign && (
          <Button
            variant="secondary"
            size="xs"
            leadingIcon={UserSwitchIcon}
            onClick={() => onReassign(card)}
          >
            Reassign…
          </Button>
        )}
        {card && columns.length > 0 && (
          <Select
            id={`task-column-${sessionId}`}
            label="Column"
            hideLabel
            size="small"
            placeholder="Move to…"
            value={columnKey ?? undefined}
            onValueChange={onMove}
            options={columns.map((column) => ({
              value: column.key,
              label: column.name,
            }))}
            wrapperClassName="mb-0"
            className="h-7 w-36 rounded-md px-2.5 text-xs"
          />
        )}
        {card && columnKey !== "done" && (
          <Button
            variant="secondary"
            size="xs"
            leadingIcon={CheckmarkCircle02Icon}
            onClick={handleMarkDone}
          >
            Mark done
          </Button>
        )}
        <div className="ml-auto flex items-center gap-1">
          <Button
            variant="toggle"
            size="xs"
            leadingIcon={Folder01Icon}
            aria-pressed={showFiles}
            onClick={onToggleFiles}
          >
            Files
          </Button>
          <Button
            as="NextLink"
            href={`/chat?sessionId=${sessionId}`}
            variant="ghost"
            size="icon-xs"
            aria-label="Expand"
          >
            <Icon icon={ArrowExpand01Icon} size={14} aria-hidden />
          </Button>
        </div>
      </div>
    </header>
  );
}
