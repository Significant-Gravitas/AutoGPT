"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import { RenameFileDialog } from "../../RenameFileDialog/RenameFileDialog";
import { useFileDrag } from "../../WorkspaceFolders/useFileDrag";
import { ExpertBadge } from "../ExpertBadge";
import { FileActionsMenu } from "../FileActionsMenu";
import { formatDayLabel, formatFileSize, formatFullDate } from "../helpers";
import { FilePreviewCard } from "./FilePreviewCard";
import { FileThumbnail } from "./FileThumbnail";
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
  file: WorkspaceFileItem;
  onOpen: (file: WorkspaceFileItem) => void;
  /** Position in the list; drives the small entrance stagger. */
  index?: number;
}

// Long enough that skimming down the list doesn't flash previews; short
// enough that a deliberate pause shows one. Moving between rows within the
// provider's skip delay opens the next preview immediately.
const PREVIEW_DELAY_MS = 450;

export function FileRow({ file, onOpen, index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const { handleDragStart, handleDragEnd } = useFileDrag(file.id, file.name);
  const [isRenameOpen, setIsRenameOpen] = useState(false);

  return (
    <motion.li
      variants={reduceMotion ? REDUCED_ROW_VARIANTS : ROW_VARIANTS}
      custom={index}
      initial={reduceMotion ? false : "hidden"}
      animate="show"
      className={cn(
        ROW_GRID_CLASS,
        "group cursor-pointer px-2 transition-colors has-[[data-state=open]]:bg-zinc-50 hover:bg-zinc-50",
      )}
      data-testid="artifacts-list-item"
      draggable
      onClick={() => onOpen(file)}
      onDragStartCapture={handleDragStart}
      onDragEndCapture={handleDragEnd}
    >
      {/* The name is the row's accessible control: it takes focus, Enter
          bubbles a click up to the row, and hovering (or focusing) it opens
          the large preview. The rename pencil sits beside it, outside the
          preview trigger. */}
      <div className="flex min-w-0 items-center gap-1 justify-self-start">
        <Tooltip delayDuration={PREVIEW_DELAY_MS}>
          <TooltipTrigger asChild>
            <button
              type="button"
              className={NAME_BUTTON_CLASS}
              data-testid="artifacts-card-open"
            >
              <FileThumbnail file={file} />
              <Text
                variant="body-medium"
                as="span"
                className="truncate text-zinc-900"
              >
                {file.name}
              </Text>
              <ExpertBadge expertId={file.expert_id} className="shrink-0" />
            </button>
          </TooltipTrigger>
          <TooltipPortal>
            {/* aria-label keeps Radix from mirroring the whole card into its
              visually-hidden tooltip copy (which would fetch previews twice). */}
            <TooltipContent
              aria-label={`Preview of ${file.name}`}
              side="right"
              align="center"
              sideOffset={16}
              collisionPadding={16}
              className="max-w-none rounded-2xl border border-zinc-200 bg-white p-0 text-sm shadow-xl shadow-black/10 outline-none"
            >
              <FilePreviewCard file={file} />
            </TooltipContent>
          </TooltipPortal>
        </Tooltip>
        <button
          type="button"
          aria-label={`Rename ${file.name}`}
          className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-zinc-400 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-zinc-100 hover:text-zinc-900 focus-visible:opacity-100"
          onClick={(e) => {
            e.stopPropagation();
            setIsRenameOpen(true);
          }}
          data-testid="artifacts-rename"
        >
          <Icon icon={PencilEdit02Icon} size={16} />
        </button>
      </div>
      <Text
        variant="body"
        as="span"
        className={cn(DATE_CELL_CLASS, "text-zinc-500")}
        title={formatFullDate(file.created_at)}
      >
        {formatDayLabel(file.created_at)}
      </Text>
      <Text
        variant="body"
        as="span"
        className={cn(SIZE_CELL_CLASS, "text-zinc-500")}
      >
        {formatFileSize(file.size_bytes)}
      </Text>
      <div
        className={cn(
          ACTIONS_CELL_CLASS,
          "opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 has-[[data-state=open]]:opacity-100",
        )}
        onClick={(e) => e.stopPropagation()}
      >
        <FileActionsMenu file={file} />
      </div>
      {isRenameOpen && (
        // The dialog portals out of the row in the DOM, but React events still
        // bubble through the component tree, so its clicks must not reach the
        // row's open handler.
        <div onClick={(e) => e.stopPropagation()}>
          <RenameFileDialog
            file={file}
            isOpen={isRenameOpen}
            setIsOpen={setIsRenameOpen}
          />
        </div>
      )}
    </motion.li>
  );
}
