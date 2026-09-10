"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Text } from "@/components/atoms/Text/Text";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import { motion, useReducedMotion } from "framer-motion";
import type { Variants } from "framer-motion";
import { EmptyState } from "../EmptyState";
import type { EmptyStateContent } from "../helpers";
import { SelectionBar } from "../SelectionBar/SelectionBar";
import { FileRow } from "./FileRow";
import { FolderRows } from "./FolderRows";
import { SkeletonRow } from "./SkeletonRow";
import { DATE_CELL_CLASS, ROW_GRID_CLASS, SIZE_CELL_CLASS } from "./row-layout";
import { useFileSelection } from "./useFileSelection";

interface Props {
  files: WorkspaceFileItem[];
  isLoading: boolean;
  emptyState: EmptyStateContent;
  compactEmpty: boolean;
  listKey: string;
  showFolders: boolean;
  onSelectFolder: (folderId: string) => void;
  onOpen: (file: WorkspaceFileItem) => void;
}

const LIST_VARIANTS: Variants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.03, delayChildren: 0.04 },
  },
};

export function ArtifactsTable({
  files,
  isLoading,
  emptyState,
  compactEmpty,
  listKey,
  showFolders,
  onSelectFolder,
  onOpen,
}: Props) {
  const reduceMotion = useReducedMotion();
  const selection = useFileSelection(files);

  return (
    <div data-testid="artifacts-table">
      {selection.selectedFiles.length > 0 ? (
        <SelectionBar
          selectedFiles={selection.selectedFiles}
          totalCount={files.length}
          onSelectAll={selection.selectAll}
          onClear={selection.clear}
        />
      ) : (
        <div className={cn(ROW_GRID_CLASS, "px-2 pb-2")}>
          <Text variant="body" as="span" className="text-zinc-500">
            Name
          </Text>
          <Text
            variant="body"
            as="span"
            className={cn(DATE_CELL_CLASS, "text-zinc-500")}
          >
            Modified
          </Text>
          <Text
            variant="body"
            as="span"
            className={cn(SIZE_CELL_CLASS, "text-zinc-500")}
          >
            Size
          </Text>
          <span aria-hidden />
        </div>
      )}
      <TooltipProvider delayDuration={450} skipDelayDuration={300}>
        <motion.ul
          key={listKey}
          className="divide-y divide-zinc-100"
          data-testid="artifacts-list"
          variants={reduceMotion ? undefined : LIST_VARIANTS}
          initial={reduceMotion ? false : "hidden"}
          animate={reduceMotion ? undefined : "show"}
        >
          {showFolders ? <FolderRows onSelectFolder={onSelectFolder} /> : null}
          {isLoading ? (
            <li
              className="divide-y divide-zinc-100"
              data-testid="artifacts-loading"
            >
              {Array.from({ length: 6 }).map((_, i) => (
                <SkeletonRow key={i} />
              ))}
            </li>
          ) : (
            files.map((file, index) => (
              <FileRow
                key={file.id}
                file={file}
                onOpen={onOpen}
                index={index}
                isSelected={selection.isSelected(file)}
                selectedIds={selection.selectedIds}
                onToggleSelect={selection.toggle}
              />
            ))
          )}
        </motion.ul>
      </TooltipProvider>
      {!isLoading && files.length === 0 ? (
        <EmptyState content={emptyState} compact={compactEmpty} />
      ) : null}
    </div>
  );
}
