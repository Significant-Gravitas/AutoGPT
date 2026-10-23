"use client";

import {
  folderSummary,
  subfolderCountOf,
} from "@/app/(platform)/artifacts/components/WorkspaceFolders/folderTree";
import { FOLDER_STYLE } from "@/app/(platform)/artifacts/components/WorkspaceFolders/folder-constants";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import {
  ArrowRight01Icon,
  CheckmarkCircle02Icon,
  Circle,
  Folder01Icon,
} from "@hugeicons/core-free-icons";

interface Props {
  folders: WorkspaceFolder[];
  allFolders: WorkspaceFolder[];
  selectedIds: ReadonlySet<string>;
  isLoading: boolean;
  onOpen: (folderId: string) => void;
  onToggleAttach: (folder: WorkspaceFolder, subfolderCount: number) => void;
}

/**
 * Folder rows above the file cards. They sit outside the cards' listbox
 * because a listbox option may not contain interactive children, and a folder
 * row has two: open, and attach.
 */
export function WorkspaceFolderRows({
  folders,
  allFolders,
  selectedIds,
  isLoading,
  onOpen,
  onToggleAttach,
}: Props) {
  if (isLoading) return <Skeleton className="h-14 w-full rounded-2xl" />;
  if (folders.length === 0) return null;

  return (
    <ul role="list" aria-label="Folders" className="flex flex-col gap-2">
      {folders.map((folder) => {
        const isSelected = selectedIds.has(folder.id);
        const subfolderCount = subfolderCountOf(allFolders, folder.id);
        return (
          <li
            key={folder.id}
            className={cn(
              "flex w-full items-center gap-3 rounded-2xl border bg-white p-3 transition-colors",
              isSelected
                ? "border-violet-300 ring-1 ring-violet-200"
                : "border-zinc-200 hover:border-zinc-300",
            )}
          >
            <button
              type="button"
              aria-label={`Open ${folder.name}`}
              onClick={() => onOpen(folder.id)}
              onKeyDown={(e) => {
                if (!isKey(e, "Enter", " ")) return;
                e.preventDefault();
                onOpen(folder.id);
              }}
              className="flex min-w-0 flex-1 items-center gap-3 rounded-xl text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-300"
            >
              <div
                className={cn(
                  "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl",
                  FOLDER_STYLE.surface,
                )}
              >
                <Icon
                  icon={Folder01Icon}
                  size={18}
                  className={FOLDER_STYLE.icon}
                />
              </div>
              <div className="flex min-w-0 flex-1 flex-col">
                <Text
                  variant="body-medium"
                  className="truncate text-zinc-900"
                  title={folder.name}
                >
                  {folder.name}
                </Text>
                <Text variant="small" className="text-zinc-500">
                  {folderSummary(folder.file_count ?? 0, subfolderCount)}
                </Text>
              </div>
            </button>
            <button
              type="button"
              aria-pressed={isSelected}
              aria-label={`Attach folder ${folder.name}`}
              onClick={(e) => {
                e.stopPropagation();
                onToggleAttach(folder, subfolderCount);
              }}
              className={cn(
                "flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-sm transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-300",
                isSelected
                  ? "text-violet-600"
                  : "text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700",
              )}
            >
              Attach
              <Icon
                icon={isSelected ? CheckmarkCircle02Icon : Circle}
                className="h-4 w-4"
              />
            </button>
            <Icon
              icon={ArrowRight01Icon}
              size={16}
              className="shrink-0 text-zinc-400"
              aria-hidden
            />
          </li>
        );
      })}
    </ul>
  );
}
