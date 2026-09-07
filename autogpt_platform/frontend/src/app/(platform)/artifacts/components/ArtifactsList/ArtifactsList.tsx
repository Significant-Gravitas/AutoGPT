"use client";

import type { TenantScope } from "../../useArtifactsFolders";

import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useState } from "react";
import type { ArtifactsView } from "../../useArtifactsPage";
import { FileViewerModal } from "../FileViewerModal/FileViewerModal";
import { WorkspaceFolders } from "../WorkspaceFolders/WorkspaceFolders";
import { ArtifactsGrid } from "./ArtifactsGrid";
import { ArtifactsTable } from "./ArtifactsTable/ArtifactsTable";
import { LoadMoreSentinel } from "./LoadMoreSentinel";

interface Props {
  scope?: TenantScope;
  files: WorkspaceFileItem[];
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  emptyMessage: string;
  compactEmpty: boolean;
  hasMore: boolean;
  isLoadingMore: boolean;
  onLoadMore: () => void;
  listKey: string;
  view: ArtifactsView;
  showFolders: boolean;
  onSelectFolder: (folder: WorkspaceFolder) => void;
}

export function ArtifactsList({
  scope,
  files,
  isLoading,
  isError,
  error,
  emptyMessage,
  compactEmpty,
  hasMore,
  isLoadingMore,
  onLoadMore,
  listKey,
  view,
  showFolders,
  onSelectFolder,
}: Props) {
  const [openFile, setOpenFile] = useState<WorkspaceFileItem | null>(null);

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

  return (
    <>
      {view === "grid" ? (
        <div className="flex flex-col gap-6">
          {showFolders ? (
            <WorkspaceFolders onSelectFolder={onSelectFolder} scope={scope} />
          ) : null}
          <ArtifactsGrid
            files={files}
            isLoading={isLoading}
            emptyMessage={emptyMessage}
            compactEmpty={compactEmpty}
            listKey={listKey}
            onOpen={setOpenFile}
          />
        </div>
      ) : (
        <ArtifactsTable
          scope={scope}
          files={files}
          isLoading={isLoading}
          emptyMessage={emptyMessage}
          compactEmpty={compactEmpty}
          listKey={listKey}
          showFolders={showFolders}
          onSelectFolder={onSelectFolder}
          onOpen={setOpenFile}
        />
      )}
      <LoadMoreSentinel
        hasMore={hasMore}
        isLoading={isLoadingMore}
        onLoadMore={onLoadMore}
        view={view}
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
