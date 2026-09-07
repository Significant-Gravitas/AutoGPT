"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
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
  Download04Icon,
  Folder01Icon,
  LinkSquare01Icon,
  Loading03Icon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { MoveToFolderDialog } from "../MoveToFolderDialog/MoveToFolderDialog";
import { useFileActionsMenu } from "./useFileActionsMenu";

interface Props {
  file: WorkspaceFileItem;
  className?: string;
}

export function FileActionsMenu({ file, className }: Props) {
  const {
    goLabel,
    goHref,
    isMoveOpen,
    setIsMoveOpen,
    isDownloading,
    isDeleting,
    handleDownload,
    handleDelete,
  } = useFileActionsMenu(file);

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            aria-label={`Actions for ${file.name}`}
            className={cn(
              "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900",
              className,
            )}
            data-testid="artifacts-card-menu"
          >
            <Icon icon={MoreHorizontalIcon} size={20} />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-44">
          <DropdownMenuItem
            onSelect={handleDownload}
            disabled={isDownloading}
            data-testid="artifacts-download"
          >
            {isDownloading ? (
              <Icon
                icon={Loading03Icon}
                size={16}
                className="mr-2 animate-spin"
              />
            ) : (
              <Icon icon={Download04Icon} size={16} className="mr-2" />
            )}
            {isDownloading ? "Downloading…" : "Download"}
          </DropdownMenuItem>
          <DropdownMenuItem asChild>
            <Link href={goHref} data-testid="artifacts-origin-link">
              <Icon icon={LinkSquare01Icon} size={16} className="mr-2" />
              {goLabel}
            </Link>
          </DropdownMenuItem>
          {/* preventDefault keeps the menu mounted while the dialog opens, so
              the menu's focus return doesn't fight the dialog for focus. */}
          <DropdownMenuItem
            onSelect={(e) => {
              e.preventDefault();
              setIsMoveOpen(true);
            }}
            data-testid="artifacts-move-to-folder"
          >
            <Icon icon={Folder01Icon} size={16} className="mr-2" />
            Move to folder
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onSelect={handleDelete}
            disabled={isDeleting}
            className="text-red-600 focus:bg-red-50 focus:text-red-700"
            data-testid="artifacts-delete"
          >
            {isDeleting ? (
              <Icon
                icon={Loading03Icon}
                size={16}
                className="mr-2 animate-spin"
              />
            ) : (
              <Icon icon={Delete02Icon} size={16} className="mr-2" />
            )}
            {isDeleting ? "Deleting…" : "Delete"}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {isMoveOpen && (
        <MoveToFolderDialog
          fileId={file.id}
          fileName={file.name}
          currentFolderId={file.folder_id}
          organizationId={file.organization_id ?? null}
          teamId={file.team_id ?? null}
          isOpen={isMoveOpen}
          setIsOpen={setIsMoveOpen}
        />
      )}
    </>
  );
}
