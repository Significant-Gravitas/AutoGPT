"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useArtifactsFolders } from "../../useArtifactsFolders";
import { FOLDER_STYLE } from "../WorkspaceFolders/folder-constants";
import { Folder01Icon, Home01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  fileId: string;
  fileName: string;
  currentFolderId?: string | null;
  organizationId: string | null;
  teamId: string | null;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

export function MoveToFolderDialog({
  fileId,
  fileName,
  currentFolderId,
  organizationId,
  teamId,
  isOpen,
  setIsOpen,
}: Props) {
  const sourceScope = { organizationId, teamId };
  const { folders, moveFileToFolder } = useArtifactsFolders(sourceScope);
  const destinationFolders = folders.filter((f) => f.id !== currentFolderId);

  function handleMove(folderId: string | null) {
    const folder =
      folderId === null
        ? null
        : (folders.find((candidate) => candidate.id === folderId) ?? null);
    // Close only on success; the hook toasts on error and we keep the dialog
    // open so the user can retry without re-opening it.
    moveFileToFolder({ fileId, folder, sourceScope })
      .then(() => setIsOpen(false))
      .catch(() => {});
  }

  return (
    <Dialog
      controlled={{ isOpen, set: setIsOpen }}
      styling={{ maxWidth: "28rem" }}
      title="Move to folder"
    >
      <Dialog.Content>
        <div className="flex flex-col gap-1">
          <Text variant="small" className="mb-1 text-zinc-500">
            Move &ldquo;{fileName}&rdquo; to:
          </Text>
          {currentFolderId != null && (
            <Button
              variant="ghost"
              className="w-full justify-start gap-3 px-3 py-2.5"
              onClick={() => handleMove(null)}
              data-testid="move-to-root"
            >
              <Icon icon={Home01Icon} size={18} className="text-zinc-500" />
              <Text variant="small-medium">Files (root)</Text>
            </Button>
          )}
          {destinationFolders.length === 0 ? (
            <div className="flex h-20 items-center justify-center">
              <Text variant="small" className="text-zinc-400">
                {folders.length === 0 ? "No folders yet" : "No other folders"}
              </Text>
            </div>
          ) : (
            destinationFolders.map((folder) => {
              const style = FOLDER_STYLE;
              return (
                <Button
                  key={folder.id}
                  variant="ghost"
                  className="w-full justify-start gap-3 px-3 py-2.5"
                  onClick={() => handleMove(folder.id)}
                  data-testid="move-to-folder-option"
                >
                  <Icon icon={Folder01Icon} size={18} className={style.icon} />
                  <Text variant="small-medium">{folder.name}</Text>
                </Button>
              );
            })
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
