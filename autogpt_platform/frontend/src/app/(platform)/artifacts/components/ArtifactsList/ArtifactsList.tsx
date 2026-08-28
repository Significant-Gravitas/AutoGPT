"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { motion, useReducedMotion } from "framer-motion";
import type { Variants } from "framer-motion";
import { useState } from "react";
import { ArtifactCard } from "./ArtifactCard/ArtifactCard";
import { FileViewerModal } from "../FileViewerModal/FileViewerModal";
import { FileTypeMarquee } from "./FileTypeMarquee";
import { LoadMoreSentinel } from "./LoadMoreSentinel";

interface Props {
  files: WorkspaceFileItem[];
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  hasSearchTerm: boolean;
  pagination: {
    hasMore: boolean;
    isLoading: boolean;
    onLoadMore: () => void;
  };
  listKey: string;
  founderView?: {
    hasHiddenTechnicalFiles: boolean;
  };
}

const GRID_VARIANTS: Variants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.04, delayChildren: 0.05 },
  },
};

export function ArtifactsList({
  files,
  isLoading,
  isError,
  error,
  hasSearchTerm,
  pagination,
  listKey,
  founderView,
}: Props) {
  const reduceMotion = useReducedMotion();
  const [openFile, setOpenFile] = useState<WorkspaceFileItem | null>(null);
  const founderMode = founderView !== undefined;

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

  if (isError) {
    return (
      <ErrorCard
        context="artifacts"
        responseError={
          error instanceof Error ? { message: error.message } : undefined
        }
      />
    );
  }

  if (files.length === 0) {
    return (
      <>
        <div
          className="flex min-h-[20rem] flex-col items-center justify-center gap-4 p-8 text-center"
          data-testid="artifacts-empty"
        >
          <FileTypeMarquee />
          <Text variant="h5" className="text-zinc-700">
            {founderView?.hasHiddenTechnicalFiles
              ? "No deliverables yet"
              : hasSearchTerm
                ? "No files match your search"
                : "No files yet"}
          </Text>
        </div>
        {founderMode ? (
          <LoadMoreSentinel
            hasMore={pagination.hasMore}
            isLoading={pagination.isLoading}
            onLoadMore={pagination.onLoadMore}
          />
        ) : null}
      </>
    );
  }

  return (
    <>
      <motion.ul
        key={listKey}
        className="grid grid-cols-1 gap-4 pt-4 sm:grid-cols-2 md:grid-cols-4"
        data-testid="artifacts-list"
        variants={reduceMotion ? undefined : GRID_VARIANTS}
        initial={reduceMotion ? false : "hidden"}
        animate={reduceMotion ? undefined : "show"}
      >
        {files.map((file) => (
          <ArtifactCard
            key={file.id}
            file={file}
            onOpen={setOpenFile}
            founderMode={founderMode}
          />
        ))}
      </motion.ul>
      <LoadMoreSentinel
        hasMore={pagination.hasMore}
        isLoading={pagination.isLoading}
        onLoadMore={pagination.onLoadMore}
      />
      {openFile ? (
        <FileViewerModal
          key={openFile.id}
          file={openFile}
          onClose={() => setOpenFile(null)}
        />
      ) : null}
    </>
  );
}
