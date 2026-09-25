"use client";

import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { useArtifactsFolders } from "../../useArtifactsFolders";
import { MoveToFolderDialog } from "../MoveToFolderDialog/MoveToFolderDialog";
import { DeleteFolderDialog } from "./DeleteFolderDialog";
import { FolderFormDialog } from "./FolderFormDialog";
import { descendantIdsOf } from "./folderTree";

interface Props {
  editing: WorkspaceFolder | null;
  moving: WorkspaceFolder | null;
  deleting: WorkspaceFolder | null;
  onEditClose: () => void;
  onMoveClose: () => void;
  onDeleteClose: () => void;
  /** Called with the deleted folder once the delete succeeds, so the page can
   *  leave a folder that no longer exists. */
  onDeleted?: (folder: WorkspaceFolder) => void;
}

// Rename + move + delete dialogs shared by the folder cards (grid view) and
// folder rows (list view). Dialogs close only on success; the hook toasts on
// error so the user's input isn't lost.
export function FolderDialogs({
  editing,
  moving,
  deleting,
  onEditClose,
  onMoveClose,
  onDeleteClose,
  onDeleted,
}: Props) {
  const { folders, updateFolder, isUpdating, deleteFolder, isDeleting } =
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
      {moving && (
        <MoveToFolderDialog
          move={{ kind: "folder", folderId: moving.id }}
          subject={`“${moving.name}”`}
          isOpen
          setIsOpen={(open) => !open && onMoveClose()}
        />
      )}
      <DeleteFolderDialog
        isOpen={deleting !== null}
        setIsOpen={(open) => !open && onDeleteClose()}
        folderName={deleting?.name ?? ""}
        subfolderCount={
          deleting ? descendantIdsOf(folders, deleting.id).size : 0
        }
        isDeleting={isDeleting}
        onConfirm={() => {
          if (!deleting) return;
          deleteFolder(deleting.id)
            .then(() => {
              onDeleted?.(deleting);
              onDeleteClose();
            })
            .catch(() => {});
        }}
      />
    </>
  );
}
