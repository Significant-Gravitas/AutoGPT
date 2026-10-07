"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  Cancel01Icon,
  Delete02Icon,
  Folder01Icon,
} from "@hugeicons/core-free-icons";
import { MoveToFolderDialog } from "../../MoveToFolderDialog/MoveToFolderDialog";
import { useSelectionBar } from "./useSelectionBar";

interface Props {
  selectedFiles: WorkspaceFileItem[];
  totalCount: number;
  onSelectAll: () => void;
  onClear: () => void;
}

export function SelectionBar({
  selectedFiles,
  totalCount,
  onSelectAll,
  onClear,
}: Props) {
  const {
    isMoveOpen,
    setIsMoveOpen,
    isDeleting,
    handleDelete,
    moveSubject,
    sharedFolderId,
  } = useSelectionBar(selectedFiles, onClear);
  const count = selectedFiles.length;

  return (
    <div
      className="flex h-7.5 items-center justify-between gap-3 px-2 pb-2"
      data-testid="artifacts-selection-bar"
    >
      <div className="flex items-center gap-3">
        <Text variant="body-medium" as="span" className="text-zinc-900">
          {count} selected
        </Text>
        {count < totalCount ? (
          <Button
            type="button"
            variant="ghost"
            className="h-auto min-w-0 rounded-none border-0 p-0 text-sm font-normal text-zinc-500 underline-offset-2 hover:bg-transparent hover:text-zinc-900 hover:underline"
            onClick={onSelectAll}
            data-testid="artifacts-select-all"
          >
            Select all {totalCount}
          </Button>
        ) : null}
      </div>
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="xs"
          onClick={() => setIsMoveOpen(true)}
          data-testid="artifacts-selection-move"
        >
          <Icon icon={Folder01Icon} size={14} />
          Move to folder
        </Button>
        <Button
          variant="outline"
          size="xs"
          disabled={isDeleting}
          onClick={handleDelete}
          className="text-red-600 hover:bg-red-50 hover:text-red-700"
          data-testid="artifacts-selection-delete"
        >
          <Icon icon={Delete02Icon} size={14} />
          {isDeleting ? "Deleting…" : "Delete"}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          withTooltip={false}
          aria-label="Clear selection"
          className="rounded-full border-0 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900"
          onClick={onClear}
          data-testid="artifacts-selection-clear"
        >
          <Icon icon={Cancel01Icon} size={16} />
        </Button>
      </div>
      {isMoveOpen && (
        <MoveToFolderDialog
          move={{
            kind: "files",
            fileIds: selectedFiles.map((file) => file.id),
            currentFolderId: sharedFolderId,
          }}
          subject={moveSubject}
          canMoveToRoot={selectedFiles.some((file) => file.folder_id != null)}
          isOpen={isMoveOpen}
          setIsOpen={setIsMoveOpen}
          onMoved={onClear}
        />
      )}
    </div>
  );
}
