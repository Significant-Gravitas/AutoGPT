"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Folder01Icon } from "@hugeicons/core-free-icons";
import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import type { DragEvent } from "react";
import { FILE_DRAG_MIME, readFileDragIds } from "../../WorkspaceFolders/drag";
import { FolderActionsMenu } from "../../WorkspaceFolders/FolderActionsMenu";
import { folderSummary } from "../../WorkspaceFolders/folderTree";
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
  subfolderCount: number;
  updatedAt: string | Date;
  onOpen: () => void;
  onEdit: () => void;
  onMove: () => void;
  onDelete: () => void;
  onFileDrop: (fileIds: string[]) => void;
  /** Position in the list; drives the small entrance stagger. */
  index?: number;
}

export function FolderRow({
  id,
  name,
  fileCount,
  subfolderCount,
  updatedAt,
  onOpen,
  onEdit,
  onMove,
  onDelete,
  onFileDrop,
  index = 0,
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
    // Animates on its own mount (see row-layout.ts) so a folder created
    // while the list is already showing doesn't stay in the hidden state.
    <motion.li
      variants={reduceMotion ? REDUCED_ROW_VARIANTS : ROW_VARIANTS}
      custom={index}
      initial={reduceMotion ? false : "hidden"}
      animate="show"
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
        {folderSummary(fileCount, subfolderCount)}
      </Text>
      <div
        className={cn(
          ACTIONS_CELL_CLASS,
          "gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100",
        )}
        onClick={(e) => e.stopPropagation()}
      >
        <FolderActionsMenu
          folderName={name}
          onRename={onEdit}
          onMove={onMove}
          onDelete={onDelete}
        />
      </div>
    </motion.li>
  );
}
