"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  FolderExportIcon,
  Home01Icon,
  KanbanIcon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { useMoveToProject } from "./useMoveToProject";

interface Props {
  cardId: string;
  triggerClassName: string;
}

function stopPropagation(event: React.SyntheticEvent) {
  event.stopPropagation();
}

export function MoveToProjectMenu({ cardId, triggerClassName }: Props) {
  const { isOpen, setIsOpen, targets, isLoading, moveTo } =
    useMoveToProject(cardId);

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label="Task actions"
          onClick={stopPropagation}
          onPointerDown={stopPropagation}
          className={triggerClassName}
        >
          <Icon icon={MoreHorizontalIcon} size={16} aria-hidden />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-48"
        onClick={stopPropagation}
      >
        <DropdownMenuSub>
          <DropdownMenuSubTrigger className="gap-2">
            <Icon icon={FolderExportIcon} size={16} aria-hidden />
            Move to project
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="max-h-80 w-52 overflow-y-auto">
            {targets.map((target) => (
              <DropdownMenuItem
                key={target.ref}
                onSelect={() => moveTo(target)}
                className="gap-2"
              >
                <Icon
                  icon={target.isHome ? Home01Icon : KanbanIcon}
                  size={16}
                  aria-hidden
                />
                <span className="truncate">{target.name}</span>
              </DropdownMenuItem>
            ))}
            {isLoading && (
              <DropdownMenuItem disabled>Loading…</DropdownMenuItem>
            )}
            {!isLoading && targets.length === 0 && (
              <DropdownMenuItem disabled>No projects yet</DropdownMenuItem>
            )}
          </DropdownMenuSubContent>
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
