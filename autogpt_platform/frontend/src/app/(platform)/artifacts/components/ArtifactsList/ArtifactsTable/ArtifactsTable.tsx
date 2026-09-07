"use client";

import type { TenantScope } from "../../../useArtifactsFolders";

import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Text } from "@/components/atoms/Text/Text";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import { motion, useReducedMotion } from "framer-motion";
import type { Variants } from "framer-motion";
import { EmptyState } from "../EmptyState";
import { FileRow } from "./FileRow";
import { FolderRows } from "./FolderRows";
import { SkeletonRow } from "./SkeletonRow";
import { DATE_CELL_CLASS, ROW_GRID_CLASS, SIZE_CELL_CLASS } from "./row-layout";

interface Props {
  scope?: TenantScope;
  files: WorkspaceFileItem[];
  isLoading: boolean;
  emptyMessage: string;
  compactEmpty: boolean;
  listKey: string;
  showFolders: boolean;
  onSelectFolder: (folder: WorkspaceFolder) => void;
  onOpen: (file: WorkspaceFileItem) => void;
}

const LIST_VARIANTS: Variants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.03, delayChildren: 0.04 },
  },
};

export function ArtifactsTable({
  scope,
  files,
  isLoading,
  emptyMessage,
  compactEmpty,
  listKey,
  showFolders,
  onSelectFolder,
  onOpen,
}: Props) {
  const reduceMotion = useReducedMotion();

  return (
    <div data-testid="artifacts-table">
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
        <span aria-hidden className="min-w-10" />
      </div>
      <TooltipProvider delayDuration={450} skipDelayDuration={300}>
        <motion.ul
          key={listKey}
          className="divide-y divide-zinc-100"
          data-testid="artifacts-list"
          variants={reduceMotion ? undefined : LIST_VARIANTS}
          initial={reduceMotion ? false : "hidden"}
          animate={reduceMotion ? undefined : "show"}
        >
          {showFolders ? (
            <FolderRows onSelectFolder={onSelectFolder} scope={scope} />
          ) : null}
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
            files.map((file) => (
              <FileRow key={file.id} file={file} onOpen={onOpen} />
            ))
          )}
        </motion.ul>
      </TooltipProvider>
      {!isLoading && files.length === 0 ? (
        <EmptyState message={emptyMessage} compact={compactEmpty} />
      ) : null}
    </div>
  );
}
