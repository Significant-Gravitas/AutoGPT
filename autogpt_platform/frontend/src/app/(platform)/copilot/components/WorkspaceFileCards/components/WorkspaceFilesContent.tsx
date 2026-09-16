"use client";

import { Download01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/ui/skeleton";
import { DeleteFileDialog } from "../../ContextPanel/components/FilesTab/components/DeleteFileDialog";
import type { SessionFile } from "../../ContextPanel/components/FilesTab/useSessionFiles";
import { WorkspaceFileCard } from "./WorkspaceFileCard";

export interface WorkspaceFilesContentProps {
  files: SessionFile[];
  isLoading: boolean;
  isError: boolean;
  isDeleting: boolean;
  isZipping: boolean;
  pendingDelete: SessionFile | null;
  onOpen: (file: SessionFile) => void;
  onDownload: (file: SessionFile) => void;
  onRequestDelete: (file: SessionFile) => void;
  onConfirmDelete: () => void;
  onCancelDelete: () => void;
  onDownloadAll: () => void;
  /** The floating stack hoists the title out of the card (see
   *  ``StackSection``); the popover keeps it inline. */
  showHeader?: boolean;
}

export function WorkspaceFilesContent({
  files,
  isLoading,
  isError,
  isDeleting,
  isZipping,
  pendingDelete,
  onOpen,
  onDownload,
  onRequestDelete,
  onConfirmDelete,
  onCancelDelete,
  onDownloadAll,
  showHeader = true,
}: WorkspaceFilesContentProps) {
  return (
    <>
      {showHeader && (
        <div className="flex items-center justify-between gap-2 pb-1.5">
          <span className="text-[15px] text-zinc-400">
            Files{files.length > 0 && ` (${files.length})`}
          </span>
          {files.length > 0 && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onDownloadAll}
              loading={isZipping}
              aria-label="Download all"
              className="size-7 rounded-lg !p-0 text-zinc-500"
            >
              <Icon icon={Download01Icon} size={15} />
            </Button>
          )}
        </div>
      )}
      <Body
        files={files}
        isLoading={isLoading}
        isError={isError}
        onOpen={onOpen}
        onDownload={onDownload}
        onRequestDelete={onRequestDelete}
      />
      <DeleteFileDialog
        fileName={pendingDelete?.item.name ?? null}
        isDeleting={isDeleting}
        onConfirm={onConfirmDelete}
        onCancel={onCancelDelete}
      />
    </>
  );
}

interface BodyProps {
  files: SessionFile[];
  isLoading: boolean;
  isError: boolean;
  onOpen: (file: SessionFile) => void;
  onDownload: (file: SessionFile) => void;
  onRequestDelete: (file: SessionFile) => void;
}

const COLLAPSED_FILE_COUNT = 4;

function Body({
  files,
  isLoading,
  isError,
  onOpen,
  onDownload,
  onRequestDelete,
}: BodyProps) {
  const [showAll, setShowAll] = useState(false);

  if (isLoading) {
    return (
      <div className="grid gap-2.5 py-1">
        <Skeleton className="h-5 w-full rounded-lg" />
        <Skeleton className="h-5 w-2/3 rounded-lg" />
      </div>
    );
  }

  if (isError) {
    return (
      <p className="py-2 text-[13px] text-zinc-400">Failed to load files.</p>
    );
  }

  if (files.length === 0) {
    return (
      <p className="py-2 text-[13px] text-zinc-400">
        No files yet. Upload one or ask Autopilot to create something.
      </p>
    );
  }

  const visibleFiles = showAll ? files : files.slice(0, COLLAPSED_FILE_COUNT);
  const hiddenCount = files.length - visibleFiles.length;

  return (
    <>
      {/* Rows bleed their hover highlight past the text with negative margins,
          so the scroller widens to match — otherwise it clips them. */}
      <div className="-mx-2.5 grid max-h-72 gap-0.5 overflow-y-auto px-2.5 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
        {visibleFiles.map((file) => (
          <WorkspaceFileCard
            key={file.item.id}
            file={file}
            onOpen={onOpen}
            onDownload={onDownload}
            onRequestDelete={onRequestDelete}
          />
        ))}
      </div>
      {files.length > COLLAPSED_FILE_COUNT && (
        <button
          type="button"
          onClick={() => setShowAll(!showAll)}
          className="w-full pb-0.5 pt-1.5 text-left text-[13px] text-zinc-500 transition-colors hover:text-zinc-800"
        >
          {showAll ? "Show less" : `View more (${hiddenCount})`}
        </button>
      )}
    </>
  );
}
