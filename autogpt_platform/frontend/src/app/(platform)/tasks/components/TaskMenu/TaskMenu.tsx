"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import { Button } from "@/components/atoms/Button/Button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  Archive02Icon,
  Copy01Icon,
  FlashOffIcon,
  Link01Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  RepeatOffIcon,
} from "@hugeicons/core-free-icons";
import { useId, type RefObject } from "react";
import { MenuRow } from "./components/MenuRow";
import { StatusSubmenu } from "./components/StatusSubmenu";
import { TaskMenuDialogs } from "./components/TaskMenuDialogs";
import { useTaskMenu } from "./useTaskMenu";

interface Props {
  task: Task;
  onRename: () => void;
  onArchived?: () => void;
  /** A copy was made with Duplicate: its id, e.g. to open it. */
  onDuplicated?: (id: string) => void;
  className?: string;
  /** -1 keeps the trigger out of the tab order, for a card the arrow keys
   *  reach instead; Tab from that card brings it back in. */
  tabIndex?: number;
}

/** The ⋯ menu on a row, a board card and the panel: start, stop or run
 *  again, rename, change the status, duplicate, copy a link, archive. */
export function TaskMenu({
  task,
  onRename,
  onArchived,
  onDuplicated,
  className,
  tabIndex,
}: Props) {
  const menu = useTaskMenu({ task, onRename, onArchived });
  const triggerID = useId();
  // Where focus goes back when the Duplicate dialog closes.
  const triggerRef: RefObject<HTMLElement | null> = {
    get current() {
      return document.getElementById(triggerID);
    },
  };

  return (
    <>
      <DropdownMenu modal={false}>
        <MenuTrigger
          task={task}
          className={className}
          tabIndex={tabIndex}
          triggerID={triggerID}
        />
        <DropdownMenuContent
          align="end"
          className="w-44"
          onCloseAutoFocus={menu.onCloseAutoFocus}
        >
          <MenuRow
            icon={menu.run.icon}
            label={menu.run.label}
            onSelect={menu.pickRun}
          />
          {task.repeat ? (
            <MenuRow
              icon={task.repeat.watch ? FlashOffIcon : RepeatOffIcon}
              label={task.repeat.watch ? "Stop watching" : "Stop repeating"}
              onSelect={menu.stopRepeating}
            />
          ) : null}
          <DropdownMenuSeparator />
          <MenuRow
            icon={PencilEdit02Icon}
            label="Rename"
            onSelect={menu.startRename}
          />
          {menu.statuses.length > 0 ? (
            <StatusSubmenu
              value={task.status}
              statuses={menu.statuses}
              onPick={menu.pickStatus}
            />
          ) : null}
          <MenuRow
            icon={Copy01Icon}
            label="Duplicate"
            onSelect={menu.startDuplicate}
          />
          <MenuRow
            icon={Link01Icon}
            label="Copy link"
            onSelect={menu.copyLink}
          />
          <DropdownMenuSeparator />
          <MenuRow
            icon={Archive02Icon}
            label="Archive"
            onSelect={menu.askArchive}
          />
        </DropdownMenuContent>
      </DropdownMenu>
      <TaskMenuDialogs
        task={task}
        menu={menu}
        triggerRef={triggerRef}
        onDuplicated={onDuplicated}
      />
    </>
  );
}

interface TriggerProps {
  task: Task;
  className?: string;
  tabIndex?: number;
  triggerID: string;
}

function MenuTrigger({ task, className, tabIndex, triggerID }: TriggerProps) {
  return (
    <DropdownMenuTrigger asChild>
      <Button
        id={triggerID}
        variant="ghost-muted"
        size="icon-xs"
        aria-label={`Actions for ${task.title}`}
        withTooltip={false}
        leadingIcon={MoreHorizontalIcon}
        className={className}
        tabIndex={tabIndex}
      />
    </DropdownMenuTrigger>
  );
}
