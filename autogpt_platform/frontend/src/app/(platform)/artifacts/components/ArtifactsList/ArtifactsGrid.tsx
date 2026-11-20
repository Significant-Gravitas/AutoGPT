"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { motion, useReducedMotion } from "framer-motion";
import type { Variants } from "framer-motion";
import { ArtifactCard } from "./ArtifactCard/ArtifactCard";
import { EmptyState } from "./EmptyState";
import type { EmptyStateContent } from "./helpers";

interface Props {
  files: WorkspaceFileItem[];
  isLoading: boolean;
  emptyState: EmptyStateContent;
  compactEmpty: boolean;
  listKey: string;
  onOpen: (file: WorkspaceFileItem) => void;
}

const GRID_VARIANTS: Variants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.04, delayChildren: 0.05 },
  },
};

export function ArtifactsGrid({
  files,
  isLoading,
  emptyState,
  compactEmpty,
  listKey,
  onOpen,
}: Props) {
  const reduceMotion = useReducedMotion();

  if (isLoading) {
    return (
      <div
        className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-4"
        data-testid="artifacts-loading"
      >
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-64 w-full rounded-2xl" />
        ))}
      </div>
    );
  }

  if (files.length === 0) {
    return <EmptyState content={emptyState} compact={compactEmpty} />;
  }

  return (
    <motion.ul
      key={listKey}
      className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-4"
      data-testid="artifacts-grid"
      variants={reduceMotion ? undefined : GRID_VARIANTS}
      initial={reduceMotion ? false : "hidden"}
      animate={reduceMotion ? undefined : "show"}
    >
      {files.map((file, index) => (
        <ArtifactCard key={file.id} file={file} onOpen={onOpen} index={index} />
      ))}
    </motion.ul>
  );
}
