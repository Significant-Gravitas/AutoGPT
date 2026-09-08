"use client";

import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useState } from "react";
import { useArtifactsFolders } from "../../../useArtifactsFolders";
import { FolderDialogs } from "../../WorkspaceFolders/FolderDialogs";
import { FolderRow } from "./FolderRow";
import { SkeletonRow } from "./SkeletonRow";

interface Props {
  onSelectFolder: (folderId: string) => void;
}

export function FolderRows({ onSelectFolder }: Props) {
  const { folders, isLoading, isError, error, moveFileToFolder } =
    useArtifactsFolders();
  const [editing, setEditing] = useState<WorkspaceFolder | null>(null);
  const [deleting, setDeleting] = useState<WorkspaceFolder | null>(null);

  if (isLoading) {
    return (
      <li data-testid="workspace-folders-loading">
        <SkeletonRow />
      </li>
    );
  }

  if (isError) {
    return (
      <li className="py-3">
        <ErrorCard
          context="folders"
          responseError={
            error instanceof Error ? { message: error.message } : undefined
          }
        />
      </li>
    );
  }

  return (
    <>
      {folders.map((folder) => (
        <FolderRow
          key={folder.id}
          id={folder.id}
          name={folder.name}
          fileCount={folder.file_count ?? 0}
          updatedAt={folder.updated_at}
          onOpen={() => onSelectFolder(folder.id)}
          onEdit={() => setEditing(folder)}
          onDelete={() => setDeleting(folder)}
          onFileDrop={(fileId) => {
            moveFileToFolder({ fileId, folderId: folder.id }).catch(() => {});
          }}
        />
      ))}
      <FolderDialogs
        editing={editing}
        deleting={deleting}
        onEditClose={() => setEditing(null)}
        onDeleteClose={() => setDeleting(null)}
      />
    </>
  );
}
