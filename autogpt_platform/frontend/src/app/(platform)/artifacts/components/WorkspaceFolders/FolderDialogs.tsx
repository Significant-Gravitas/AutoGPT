"use client";

import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { useArtifactsFolders } from "../../useArtifactsFolders";
import { DeleteFolderDialog } from "./DeleteFolderDialog";
import { FolderFormDialog } from "./FolderFormDialog";

interface Props {
  editing: WorkspaceFolder | null;
  deleting: WorkspaceFolder | null;
  onEditClose: () => void;
  onDeleteClose: () => void;
}

// Rename + delete dialogs shared by the folder cards (grid view) and folder
// rows (list view). Dialogs close only on success; the hook toasts on error
// so the user's input isn't lost.
export function FolderDialogs({
  editing,
  deleting,
  onEditClose,
  onDeleteClose,
}: Props) {
  const { updateFolder, isUpdating, deleteFolder, isDeleting } =
    useArtifactsFolders();

  return (
    <>
      <FolderFormDialog
        isOpen={editing !== null}
        setIsOpen={(open) => !open && onEditClose()}
        mode="edit"
        initialName={editing?.name}
        isSubmitting={isUpdating}
        onSubmit={(values) => {
          if (!editing) return;
          updateFolder({ folderId: editing.id, ...values })
            .then(onEditClose)
            .catch(() => {});
        }}
      />
      <DeleteFolderDialog
        isOpen={deleting !== null}
        setIsOpen={(open) => !open && onDeleteClose()}
        folderName={deleting?.name ?? ""}
        isDeleting={isDeleting}
        onConfirm={() => {
          if (!deleting) return;
          deleteFolder(deleting.id)
            .then(onDeleteClose)
            .catch(() => {});
        }}
      />
    </>
  );
}
