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
import { useState } from "react";

type Action = "rename" | "add" | "delete";

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
  const [chosen, setChosen] = useState<Action | null>(null);
  const isCustom = !column.builtin;

  // Runs once the menu has closed and let go of focus, so the rename input
  // or the confirm dialog that the action opens keeps it.
  function handleCloseAutoFocus(event: Event) {
    if (!chosen) return;
    event.preventDefault();
    setChosen(null);
    if (chosen === "rename") onRename();
    if (chosen === "add") onAddAfter();
    if (chosen === "delete") onDelete();
  }

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
        <DropdownMenuItem onSelect={() => setChosen("rename")}>
          <Icon icon={PencilEdit02Icon} size={16} />
          Rename
        </DropdownMenuItem>
        {canAddColumn && (
          <DropdownMenuItem onSelect={() => setChosen("add")}>
            <Icon icon={PlusSignIcon} size={16} />
            Add column after
          </DropdownMenuItem>
        )}
        {isCustom && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={() => setChosen("delete")}
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
