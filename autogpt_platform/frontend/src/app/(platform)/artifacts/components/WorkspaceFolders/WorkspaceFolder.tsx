"use client";

import { useState } from "react";
import { FolderIcon, PencilSimpleIcon, TrashIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { FILE_DRAG_MIME } from "./drag";
import { FOLDER_STYLE } from "./folder-constants";

interface Props {
  id: string;
  name: string;
  fileCount: number;
  onEdit: () => void;
  onDelete: () => void;
  onClick: () => void;
  onFileDrop: (fileId: string, folderId: string) => void;
}

export function WorkspaceFolder({
  id,
  name,
  fileCount,
  onEdit,
  onDelete,
  onClick,
  onFileDrop,
}: Props) {
  const [isDragOver, setIsDragOver] = useState(false);
  const style = FOLDER_STYLE;

  function handleDragOver(e: React.DragEvent<HTMLDivElement>) {
    if (e.dataTransfer.types.includes(FILE_DRAG_MIME)) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      setIsDragOver(true);
    }
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragOver(false);
    const fileId = e.dataTransfer.getData(FILE_DRAG_MIME);
    if (fileId) onFileDrop(fileId, id);
  }

  return (
    <div
      role="button"
      tabIndex={0}
      data-testid="workspace-folder"
      data-folder-id={id}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
      onDragOver={handleDragOver}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      className={`group flex cursor-pointer items-center gap-3 rounded-2xl border border-zinc-200 bg-white p-3 transition-all hover:border-zinc-300 ${
        isDragOver ? `${style.surface} ring-2 ${style.ring}` : ""
      }`}
    >
      <div
        className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${style.surface}`}
      >
        <FolderIcon size={22} weight="fill" className={style.icon} />
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        <Text
          variant="body-medium"
          className="truncate text-zinc-900"
          data-testid="workspace-folder-name"
          title={name}
        >
          {name}
        </Text>
        <Text variant="small" className="text-zinc-500">
          {fileCount} {fileCount === 1 ? "file" : "files"}
        </Text>
      </div>
      <div className="flex items-center gap-1 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
        <Button
          variant="icon"
          size="icon"
          aria-label="Rename folder"
          onClick={(e) => {
            e.stopPropagation();
            onEdit();
          }}
          className="h-9 w-9 !p-2 text-zinc-500 hover:text-zinc-800"
        >
          <PencilSimpleIcon size={16} />
        </Button>
        <Button
          variant="icon"
          size="icon"
          aria-label="Delete folder"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className="h-9 w-9 !p-2 text-zinc-500 hover:text-red-600"
        >
          <TrashIcon size={16} />
        </Button>
      </div>
    </div>
  );
}
