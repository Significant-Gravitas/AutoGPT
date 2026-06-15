"use client";

import { useState } from "react";
import { FolderSimplePlusIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import type { WorkspaceFolder as WorkspaceFolderModel } from "@/app/api/__generated__/models/workspaceFolder";
import { useArtifactsFolders } from "../../useArtifactsFolders";
import { DeleteFolderDialog } from "./DeleteFolderDialog";
import { FolderFormDialog } from "./FolderFormDialog";
import { WorkspaceFolder } from "./WorkspaceFolder";

interface Props {
  onSelectFolder: (folderId: string) => void;
}

export function WorkspaceFolders({ onSelectFolder }: Props) {
  const {
    folders,
    isLoading,
    isError,
    error,
    createFolder,
    isCreating,
    updateFolder,
    isUpdating,
    deleteFolder,
    isDeleting,
    moveFileToFolder,
  } = useArtifactsFolders();

  const [isCreateOpen, setIsCreateOpen] = useState(false);
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

  return (
    <div className="flex flex-col gap-3" data-testid="workspace-folders">
      <div className="flex items-center justify-between">
        <Text variant="large-medium" className="text-zinc-700">
          Folders
        </Text>
        <Button
          variant="secondary"
          size="small"
          onClick={() => setIsCreateOpen(true)}
          data-testid="create-folder-button"
        >
          <FolderSimplePlusIcon size={18} />
          New folder
        </Button>
      </div>

      {folders.length === 0 ? (
        <Text variant="small" className="text-zinc-400">
          No folders yet. Create one to organize your files.
        </Text>
      ) : (
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
              onFileDrop={(fileId, folderId) =>
                moveFileToFolder({ fileId, folderId })
              }
            />
          ))}
        </div>
      )}

      <FolderFormDialog
        isOpen={isCreateOpen}
        setIsOpen={setIsCreateOpen}
        mode="create"
        isSubmitting={isCreating}
        onSubmit={(values) => {
          // Close only on success; the hook toasts on error and we keep the
          // dialog open so the user's input isn't lost.
          createFolder(values)
            .then(() => setIsCreateOpen(false))
            .catch(() => {});
        }}
      />

      <FolderFormDialog
        isOpen={editing !== null}
        setIsOpen={(open) => !open && setEditing(null)}
        mode="edit"
        initialName={editing?.name}
        isSubmitting={isUpdating}
        onSubmit={(values) => {
          if (!editing) return;
          updateFolder({ folderId: editing.id, ...values })
            .then(() => setEditing(null))
            .catch(() => {});
        }}
      />

      <DeleteFolderDialog
        isOpen={deleting !== null}
        setIsOpen={(open) => !open && setDeleting(null)}
        folderName={deleting?.name ?? ""}
        isDeleting={isDeleting}
        onConfirm={() => {
          if (!deleting) return;
          deleteFolder(deleting.id)
            .then(() => setDeleting(null))
            .catch(() => {});
        }}
      />
    </div>
  );
}
