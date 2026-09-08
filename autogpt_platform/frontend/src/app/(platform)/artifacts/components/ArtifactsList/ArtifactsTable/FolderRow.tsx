"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
  Folder01Icon,
  PencilIcon,
} from "@hugeicons/core-free-icons";
import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import type { DragEvent } from "react";
import { FILE_DRAG_MIME, readFileDragIds } from "../../WorkspaceFolders/drag";
import { FOLDER_STYLE } from "../../WorkspaceFolders/folder-constants";
import { formatDayLabel, formatFullDate } from "../helpers";
import {
  ACTIONS_CELL_CLASS,
  DATE_CELL_CLASS,
  NAME_BUTTON_CLASS,
  REDUCED_ROW_VARIANTS,
  ROW_GRID_CLASS,
  ROW_VARIANTS,
  SIZE_CELL_CLASS,
} from "./row-layout";

interface Props {
  id: string;
  name: string;
  fileCount: number;
  updatedAt: string | Date;
  onOpen: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onFileDrop: (fileIds: string[]) => void;
}

const ACTION_BUTTON_CLASS =
  "inline-flex h-8 w-8 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-100";

export function FolderRow({
  id,
  name,
  fileCount,
  updatedAt,
  onOpen,
  onEdit,
  onDelete,
  onFileDrop,
}: Props) {
  const reduceMotion = useReducedMotion();
  const [isDragOver, setIsDragOver] = useState(false);

  function handleDragOver(e: DragEvent<HTMLLIElement>) {
    if (e.dataTransfer.types.includes(FILE_DRAG_MIME)) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      setIsDragOver(true);
    }
  }

  function handleDragLeave(e: DragEvent<HTMLLIElement>) {
    // Ignore leave events fired while moving onto a child element — only clear
    // the highlight when the cursor actually exits the row.
    if (e.currentTarget.contains(e.relatedTarget as Node | null)) return;
    setIsDragOver(false);
  }

  function handleDrop(e: DragEvent<HTMLLIElement>) {
    e.preventDefault();
    setIsDragOver(false);
    const fileIds = readFileDragIds(e.dataTransfer);
    if (fileIds.length > 0) onFileDrop(fileIds);
  }

  return (
    <motion.li
      variants={reduceMotion ? REDUCED_ROW_VARIANTS : ROW_VARIANTS}
      className={cn(
        ROW_GRID_CLASS,
        "group cursor-pointer px-2 transition-colors hover:bg-zinc-50",
        isDragOver && `${FOLDER_STYLE.surface} ring-2 ${FOLDER_STYLE.ring}`,
      )}
      data-folder-id={id}
      onClick={onOpen}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <button
        type="button"
        className={NAME_BUTTON_CLASS}
        data-testid="workspace-folder"
      >
        <div
          className={cn(
            "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-zinc-200",
            FOLDER_STYLE.surface,
          )}
        >
          <Icon icon={Folder01Icon} size={20} className={FOLDER_STYLE.icon} />
        </div>
        <Text
          variant="body-medium"
          as="span"
          className="truncate text-zinc-900"
          data-testid="workspace-folder-name"
        >
          {name}
        </Text>
      </button>
      <Text
        variant="body"
        as="span"
        className={cn(DATE_CELL_CLASS, "text-zinc-500")}
        title={formatFullDate(updatedAt)}
      >
        {formatDayLabel(updatedAt)}
      </Text>
      <Text
        variant="body"
        as="span"
        className={cn(SIZE_CELL_CLASS, "text-zinc-500")}
      >
        {fileCount} {fileCount === 1 ? "file" : "files"}
      </Text>
      <div
        className={cn(
          ACTIONS_CELL_CLASS,
          "gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100",
        )}
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          aria-label="Rename folder"
          onClick={onEdit}
          className={cn(ACTION_BUTTON_CLASS, "hover:text-zinc-900")}
        >
          <Icon icon={PencilIcon} size={16} />
        </button>
        <button
          type="button"
          aria-label="Delete folder"
          onClick={onDelete}
          className={cn(ACTION_BUTTON_CLASS, "hover:text-red-600")}
        >
          <Icon icon={Delete02Icon} size={16} />
        </button>
      </div>
    </motion.li>
  );
}
