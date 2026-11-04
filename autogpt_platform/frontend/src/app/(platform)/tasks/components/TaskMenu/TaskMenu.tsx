"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  Delete02Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
} from "@hugeicons/core-free-icons";
import { DeleteTaskDialog } from "./components/DeleteTaskDialog/DeleteTaskDialog";
import { useTaskMenu } from "./useTaskMenu";

interface Props {
  taskId: string;
  title: string;
  onRename: () => void;
  onDeleted?: () => void;
  className?: string;
}

/** The ⋯ menu on a row, a board card and the panel: rename, delete. */
export function TaskMenu({
  taskId,
  title,
  onRename,
  onDeleted,
  className,
}: Props) {
  const menu = useTaskMenu({ taskId, onRename, onDeleted });

  return (
    <>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost-muted"
            size="icon-xs"
            aria-label="Task actions"
            withTooltip={false}
            leadingIcon={MoreHorizontalIcon}
            className={className}
          />
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          className="w-40"
          onCloseAutoFocus={menu.onCloseAutoFocus}
        >
          <DropdownMenuItem onSelect={menu.startRename}>
            <span className="flex items-center gap-2">
              <Icon icon={PencilEdit02Icon} size={14} />
              Rename
            </span>
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={menu.askDelete}>
            <span className="flex items-center gap-2 text-error">
              <Icon icon={Delete02Icon} size={14} />
              Delete
            </span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <DeleteTaskDialog
        isOpen={menu.isConfirmingDelete}
        title={title}
        onCancel={menu.keep}
        onConfirm={menu.confirmDelete}
      />
    </>
  );
}
