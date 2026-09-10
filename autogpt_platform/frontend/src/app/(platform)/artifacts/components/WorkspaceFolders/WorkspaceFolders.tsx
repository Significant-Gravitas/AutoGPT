"use client";

import { useState } from "react";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import type { WorkspaceFolder as WorkspaceFolderModel } from "@/app/api/__generated__/models/workspaceFolder";
import { useArtifactsFolders } from "../../useArtifactsFolders";
import { FolderDialogs } from "./FolderDialogs";
import { WorkspaceFolder } from "./WorkspaceFolder";

interface Props {
  onSelectFolder: (folderId: string) => void;
}

export function WorkspaceFolders({ onSelectFolder }: Props) {
  const { folders, isLoading, isError, error, moveFileToFolder } =
    useArtifactsFolders();

  const [editing, setEditing] = useState<WorkspaceFolderModel | null>(null);
  const [deleting, setDeleting] = useState<WorkspaceFolderModel | null>(null);

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-16 w-full rounded-2xl" />
        ))}
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorCard
        context="folders"
        responseError={
          error instanceof Error ? { message: error.message } : undefined
        }
      />
    );
  }

  if (folders.length === 0) return null;

  return (
    <div className="flex flex-col gap-3" data-testid="workspace-folders">
      <Text variant="large-medium" className="text-zinc-700">
        Folders
      </Text>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
        {folders.map((folder) => (
          <WorkspaceFolder
            key={folder.id}
            id={folder.id}
            name={folder.name}
            fileCount={folder.file_count ?? 0}
            onClick={() => onSelectFolder(folder.id)}
            onEdit={() => setEditing(folder)}
            onDelete={() => setDeleting(folder)}
            onFileDrop={(fileId, folderId) => {
              moveFileToFolder({ fileId, folderId }).catch(() => {});
            }}
          />
        ))}
      </div>
      <FolderDialogs
        editing={editing}
        deleting={deleting}
        onEditClose={() => setEditing(null)}
        onDeleteClose={() => setDeleting(null)}
      />
    </div>
  );
}
