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
      className="flex h-[1.875rem] items-center justify-between gap-3 px-2 pb-2"
      data-testid="artifacts-selection-bar"
    >
      <div className="flex items-center gap-3">
        <Text variant="body-medium" as="span" className="text-zinc-900">
          {count} selected
        </Text>
        {count < totalCount ? (
          <button
            type="button"
            className="text-sm text-zinc-500 underline-offset-2 hover:text-zinc-900 hover:underline"
            onClick={onSelectAll}
            data-testid="artifacts-select-all"
          >
            Select all {totalCount}
          </button>
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
        <button
          type="button"
          aria-label="Clear selection"
          className="inline-flex h-7 w-7 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900"
          onClick={onClear}
          data-testid="artifacts-selection-clear"
        >
          <Icon icon={Cancel01Icon} size={16} />
        </button>
      </div>
      {isMoveOpen && (
        <MoveToFolderDialog
          fileIds={selectedFiles.map((file) => file.id)}
          subject={moveSubject}
          currentFolderId={sharedFolderId}
          isOpen={isMoveOpen}
          setIsOpen={setIsMoveOpen}
          onMoved={onClear}
        />
      )}
    </div>
  );
}
