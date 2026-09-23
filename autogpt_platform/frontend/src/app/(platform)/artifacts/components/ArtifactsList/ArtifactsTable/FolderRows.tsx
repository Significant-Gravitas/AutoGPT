"use client";

import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useState } from "react";
import { useArtifactsFolders } from "../../../useArtifactsFolders";
import { FolderDialogs } from "../../WorkspaceFolders/FolderDialogs";
import {
  childrenOf,
  subfolderCountOf,
} from "../../WorkspaceFolders/folderTree";
import { FolderRow } from "./FolderRow";
import { SkeletonRow } from "./SkeletonRow";

interface Props {
  /** Folder whose children are listed; `null` is the workspace root. */
  parentId: string | null;
  onSelectFolder: (folderId: string) => void;
}

export function FolderRows({ parentId, onSelectFolder }: Props) {
  const { folders, isLoading, isError, error, moveFilesToFolder } =
    useArtifactsFolders();
  const [editing, setEditing] = useState<WorkspaceFolder | null>(null);
  const [moving, setMoving] = useState<WorkspaceFolder | null>(null);
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
      {childrenOf(folders, parentId).map((folder, index) => (
        <FolderRow
          key={folder.id}
          index={index}
          id={folder.id}
          name={folder.name}
          fileCount={folder.file_count ?? 0}
          subfolderCount={subfolderCountOf(folders, folder.id)}
          updatedAt={folder.updated_at}
          onOpen={() => onSelectFolder(folder.id)}
          onEdit={() => setEditing(folder)}
          onMove={() => setMoving(folder)}
          onDelete={() => setDeleting(folder)}
          onFileDrop={(fileIds) => {
            moveFilesToFolder({ fileIds, folderId: folder.id }).catch(() => {});
          }}
        />
      ))}
      <FolderDialogs
        editing={editing}
        moving={moving}
        deleting={deleting}
        onEditClose={() => setEditing(null)}
        onMoveClose={() => setMoving(null)}
        onDeleteClose={() => setDeleting(null)}
      />
    </>
  );
}
