"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  Archive02Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  RepeatOffIcon,
  Tick02Icon,
  FlashOffIcon,
} from "@hugeicons/core-free-icons";
import { STATUS_ORDER, statusOf } from "../../helpers";
import { StopTaskDialog } from "../StopTaskDialog/StopTaskDialog";
import { TryAgainDialog } from "../TryAgainForm/TryAgainDialog";
import { ArchiveTaskDialog } from "./components/ArchiveTaskDialog/ArchiveTaskDialog";
import { useTaskMenu } from "./useTaskMenu";

interface Props {
  task: Task;
  onRename: () => void;
  onArchived?: () => void;
  className?: string;
}

/** The ⋯ menu on a row, a board card and the panel: start, stop or run
 *  again, rename, change the status, archive. */
export function TaskMenu({ task, onRename, onArchived, className }: Props) {
  const menu = useTaskMenu({ task, onRename, onArchived });

  return (
    <>
      <DropdownMenu modal={false}>
        <MenuTrigger task={task} className={className} />
        <DropdownMenuContent
          align="end"
          className="w-44"
          onCloseAutoFocus={menu.onCloseAutoFocus}
        >
          <DropdownMenuItem onSelect={menu.pickRun}>
            <span className="flex items-center gap-2">
              <Icon icon={menu.run.icon} size={14} />
              {menu.run.label}
            </span>
          </DropdownMenuItem>
          {task.repeat ? (
            <DropdownMenuItem onSelect={menu.stopRepeating}>
              <span className="flex items-center gap-2">
                <Icon
                  icon={task.repeat.watch ? FlashOffIcon : RepeatOffIcon}
                  size={14}
                />
                {task.repeat.watch ? "Stop watching" : "Stop repeating"}
              </span>
            </DropdownMenuItem>
          ) : null}
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={menu.startRename}>
            <span className="flex items-center gap-2">
              <Icon icon={PencilEdit02Icon} size={14} />
              Rename
            </span>
          </DropdownMenuItem>
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>
              <span className="flex items-center gap-2">
                <Icon icon={Tick02Icon} size={14} />
                Change status
              </span>
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent className="w-40">
              <DropdownMenuRadioGroup
                value={task.status}
                onValueChange={menu.pickStatus}
              >
                {STATUS_ORDER.map((status) => (
                  <DropdownMenuRadioItem key={status} value={status}>
                    {statusOf(status).word}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={menu.askArchive}>
            <span className="flex items-center gap-2">
              <Icon icon={Archive02Icon} size={14} />
              Archive
            </span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <ArchiveTaskDialog
        isOpen={menu.isConfirmingArchive}
        title={task.title}
        isPending={menu.isPending}
        onCancel={menu.keep}
        onConfirm={menu.confirmArchive}
      />
      <StopTaskDialog
        isOpen={menu.isConfirmingStop}
        title={task.title}
        isPending={menu.isStopping}
        onCancel={menu.closeRunDialog}
        onConfirm={menu.confirmStop}
        isRepeating={!!task.repeat}
      />
      <TryAgainDialog
        isOpen={menu.isAskingRetry}
        taskID={task.id}
        isPending={menu.isStarting}
        onCancel={menu.closeRunDialog}
        onSubmit={menu.retry}
      />
    </>
  );
}

interface TriggerProps {
  task: Task;
  className?: string;
}

function MenuTrigger({ task, className }: TriggerProps) {
  return (
    <DropdownMenuTrigger asChild>
      <Button
        variant="ghost-muted"
        size="icon-xs"
        aria-label={`Actions for ${task.title}`}
        withTooltip={false}
        leadingIcon={MoreHorizontalIcon}
        className={className}
      />
    </DropdownMenuTrigger>
  );
}
