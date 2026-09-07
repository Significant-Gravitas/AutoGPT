"use client";

import type { TenantScope } from "../../../useArtifactsFolders";

import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useState } from "react";
import { useArtifactsFolders } from "../../../useArtifactsFolders";
import { FolderDialogs } from "../../WorkspaceFolders/FolderDialogs";
import { FolderRow } from "./FolderRow";
import { SkeletonRow } from "./SkeletonRow";

interface Props {
  scope?: TenantScope;
  onSelectFolder: (folder: WorkspaceFolder) => void;
}

export function FolderRows({ onSelectFolder, scope }: Props) {
  const { folders, isLoading, isError, error, moveFileToFolder } =
    useArtifactsFolders(scope);
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
          onOpen={() => onSelectFolder(folder)}
          onEdit={() => setEditing(folder)}
          onDelete={() => setDeleting(folder)}
          onFileDrop={(fileId, sourceScope) => {
            moveFileToFolder({ fileId, folder, sourceScope }).catch(() => {});
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
