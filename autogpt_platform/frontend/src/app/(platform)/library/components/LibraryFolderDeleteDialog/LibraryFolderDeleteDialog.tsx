"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  useDeleteV2DeleteFolder,
  getGetV2ListLibraryFoldersQueryKey,
} from "@/app/api/__generated__/endpoints/folders/folders";
import { getGetV2ListLibraryAgentsQueryKey } from "@/app/api/__generated__/endpoints/library/library";
import { useQueryClient } from "@tanstack/react-query";
import type { LibraryFolder } from "@/app/api/__generated__/models/libraryFolder";

interface Props {
  folder: LibraryFolder;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  onDeleted?: () => void;
}

export function LibraryFolderDeleteDialog({
  folder,
  isOpen,
  setIsOpen,
  onDeleted,
}: Props) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const { mutate: deleteFolder, isPending } = useDeleteV2DeleteFolder({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryFoldersQueryKey(),
        });
        queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryAgentsQueryKey(),
        });
        toast({
          title: "Folder deleted",
          description: `"${folder.name}" has been deleted.`,
        });
        setIsOpen(false);
        onDeleted?.();
      },
      onError: () => {
        toast({
          title: "Error",
          description: "Failed to delete folder. Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  function handleDelete() {
    deleteFolder({ folderId: folder.id });
  }

  return (
    <Dialog
      controlled={{
        isOpen,
        set: setIsOpen,
      }}
      styling={{ maxWidth: "32rem" }}
      title="Delete folder"
    >
      <Dialog.Content>
        <div>
          <Text variant="large">
            Are you sure you want to delete &ldquo;{folder.name}&rdquo;? Agents
            inside this folder will be moved back to your library.
          </Text>
          <Dialog.Footer>
            <Button
              variant="secondary"
              disabled={isPending}
              onClick={() => setIsOpen(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              loading={isPending}
            >
              Delete Folder
            </Button>
          </Dialog.Footer>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
