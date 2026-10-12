"use client";

import { ContextPanelToggle } from "@/app/(platform)/copilot/components/ContextPanel/ContextPanelToggle";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { SheetClose, SheetTitle } from "@/components/ui/sheet";
import {
  ArrowExpand01Icon,
  Cancel01Icon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import type { Card } from "../../../../board-helpers";
import { AssigneePicker } from "../../../AssigneePicker/AssigneePicker";
import { AssigneeButton } from "./components/AssigneeButton/AssigneeButton";
import { StatusSelect } from "./components/StatusSelect/StatusSelect";
import { useTaskPanelHeader } from "./useTaskPanelHeader";

interface Props {
  sessionId: string;
  title: string;
  card: Card | null;
  columns: BoardColumn[];
  columnKey: string | null;
  showChatControls: boolean;
  onRename: (title: string) => void;
  onAssign: (expertId: string | null) => void;
  onMove: (column: string) => void;
  onReassign: (card: Card) => void;
}

const ICON_BUTTON_CLASS =
  "flex size-8 shrink-0 items-center justify-center rounded-md text-zinc-600 transition-colors hover:bg-zinc-100 hover:text-zinc-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800";

export function TaskPanelHeader({
  sessionId,
  title,
  card,
  columns,
  columnKey,
  showChatControls,
  onRename,
  onAssign,
  onMove,
  onReassign,
}: Props) {
  const {
    isEditing,
    assignee,
    canChangeAssignee,
    canReassign,
    draft,
    startEditing,
    stopEditing,
    handleChange,
    handleKeyDown,
  } = useTaskPanelHeader({ title, card, onRename });
  const chatExpert = useCopilotUIStore((s) => s.contextPanelExpert);

  function handleMarkDone() {
    onMove("done");
  }

  function handleReassign() {
    if (card) onReassign(card);
  }

  return (
    <TooltipProvider>
      <header className="relative z-10 flex h-14 shrink-0 items-center gap-2 bg-white pl-3 pr-2 shadow-[0_1px_0_rgb(0_0_0/0.05),0_8px_16px_-12px_rgb(0_0_0/0.2)]">
        {card && canChangeAssignee ? (
          <AssigneePicker
            expertId={card.expert_id ?? null}
            onSelect={onAssign}
            compact
          />
        ) : (
          <AssigneeButton
            assignee={assignee}
            onReassign={canReassign ? handleReassign : undefined}
          />
        )}

        <div className="flex min-w-0 flex-1 items-center">
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
            <SheetTitle className="min-w-0 text-sm font-semibold text-zinc-900">
              <button
                type="button"
                onClick={startEditing}
                title="Rename"
                className="max-w-full truncate rounded-md px-1.5 py-1 text-left hover:bg-zinc-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800"
              >
                {title}
              </button>
            </SheetTitle>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-1">
          {card && columns.length > 0 && (
            <StatusSelect
              sessionId={sessionId}
              columns={columns}
              columnKey={columnKey}
              onMove={onMove}
            />
          )}
          {card && columnKey !== "done" && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-label="Mark done"
                  onClick={handleMarkDone}
                  className={ICON_BUTTON_CLASS}
                >
                  <Icon icon={CheckmarkCircle02Icon} size={16} aria-hidden />
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom">Mark done</TooltipContent>
            </Tooltip>
          )}
          {showChatControls && (
            <ContextPanelToggle
              sessionId={sessionId}
              expert={chatExpert}
              showComputer={false}
              className="m-0 border-0 p-0 shadow-none"
            />
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                as="NextLink"
                href={`/chat?sessionId=${sessionId}`}
                variant="ghost"
                size="icon-xs"
                aria-label="Expand"
                className="size-8 text-zinc-600"
              >
                <Icon icon={ArrowExpand01Icon} size={14} aria-hidden />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom">Open full chat</TooltipContent>
          </Tooltip>
          <span aria-hidden className="mx-1 h-5 w-px bg-zinc-200" />
          <SheetClose aria-label="Close" className={ICON_BUTTON_CLASS}>
            <Icon icon={Cancel01Icon} size={16} aria-hidden />
          </SheetClose>
        </div>
      </header>
    </TooltipProvider>
  );
}
