"use client";

import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  Delete02Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { useColumnMenu } from "./useColumnMenu";

interface Props {
  column: BoardColumn;
  canAddColumn: boolean;
  onRename: () => void;
  onAddAfter: () => void;
  onDelete: () => void;
}

export function ColumnMenu({
  column,
  canAddColumn,
  onRename,
  onAddAfter,
  onDelete,
}: Props) {
  const { handleCloseAutoFocus, chooseRename, chooseAdd, chooseDelete } =
    useColumnMenu({ onRename, onAddAfter, onDelete });
  const isCustom = !column.builtin;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label={`${column.name} column actions`}
          className="text-zinc-500 hover:bg-zinc-200/70 hover:text-zinc-800"
        >
          <Icon icon={MoreHorizontalIcon} size={16} />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-48"
        onCloseAutoFocus={handleCloseAutoFocus}
      >
        <DropdownMenuItem onSelect={chooseRename}>
          <Icon icon={PencilEdit02Icon} size={16} />
          Rename
        </DropdownMenuItem>
        {canAddColumn && (
          <DropdownMenuItem onSelect={chooseAdd}>
            <Icon icon={PlusSignIcon} size={16} />
            Add column after
          </DropdownMenuItem>
        )}
        {isCustom && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={chooseDelete}
              className="text-red-600 focus:text-red-700"
            >
              <Icon icon={Delete02Icon} size={16} />
              Delete
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
