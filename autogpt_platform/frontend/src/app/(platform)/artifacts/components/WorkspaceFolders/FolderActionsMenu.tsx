"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
  Folder01Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
} from "@hugeicons/core-free-icons";

interface Props {
  folderName: string;
  onRename: () => void;
  onMove: () => void;
  onDelete: () => void;
  className?: string;
}

export function FolderActionsMenu({
  folderName,
  onRename,
  onMove,
  onDelete,
  className,
}: Props) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label={`Actions for ${folderName}`}
          onClick={(e) => e.stopPropagation()}
          className={cn(
            "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900",
            className,
          )}
          data-testid="folder-actions-menu"
        >
          <Icon icon={MoreHorizontalIcon} size={20} />
        </button>
      </DropdownMenuTrigger>
      {/* preventDefault keeps the menu mounted while the dialog opens, so the
          menu's focus return doesn't fight the dialog for focus. */}
      <DropdownMenuContent align="end" className="w-44">
        <DropdownMenuItem
          onSelect={(e) => {
            e.preventDefault();
            onRename();
          }}
          data-testid="folder-rename-menu"
        >
          <Icon icon={PencilEdit02Icon} size={16} className="mr-2" />
          Rename
        </DropdownMenuItem>
        <DropdownMenuItem
          onSelect={(e) => {
            e.preventDefault();
            onMove();
          }}
          data-testid="folder-move-menu"
        >
          <Icon icon={Folder01Icon} size={16} className="mr-2" />
          Move to folder
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onSelect={(e) => {
            e.preventDefault();
            onDelete();
          }}
          className="text-red-600 focus:bg-red-50 focus:text-red-700"
          data-testid="folder-delete-menu"
        >
          <Icon icon={Delete02Icon} size={16} className="mr-2" />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
