import { deleteWorkspaceFile } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { invalidateWorkspaceFileQueries } from "../helpers";

export function useSelectionBar(
  selectedFiles: WorkspaceFileItem[],
  onDone: () => void,
) {
  const [isMoveOpen, setIsMoveOpen] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const count = selectedFiles.length;

  const { mutateAsync: deleteFiles, isPending: isDeleting } = useMutation({
    // There is no bulk delete endpoint, so delete one by one and report how
    // many failed; the ones that succeeded leave the list on refetch.
    mutationFn: async (fileIds: string[]) => {
      const results = await Promise.allSettled(
        fileIds.map((fileId) => deleteWorkspaceFile(fileId)),
      );
      const failed = results.filter((r) => r.status === "rejected").length;
      if (failed > 0) {
        throw new Error(
          `${failed} of ${fileIds.length} files could not be deleted.`,
        );
      }
    },
    onSettled: () => invalidateWorkspaceFileQueries(queryClient),
    onSuccess: (_, fileIds) => {
      toast({
        title:
          fileIds.length === 1
            ? "File deleted"
            : `${fileIds.length} files deleted`,
      });
      onDone();
    },
    onError: (error) => {
      toast({
        title: "Failed to delete files",
        description:
          error instanceof Error ? error.message : "Please try again.",
        variant: "destructive",
      });
    },
  });

  async function handleDelete() {
    if (isDeleting || count === 0) return;
    const confirmed = window.confirm(
      count === 1
        ? `Delete "${selectedFiles[0].name}"?`
        : `Delete ${count} files?`,
    );
    if (!confirmed) return;
    try {
      await deleteFiles(selectedFiles.map((file) => file.id));
    } catch {
      // Already surfaced by the mutation's onError toast.
    }
  }

  const firstFolderId = selectedFiles[0]?.folder_id ?? null;
  const sharedFolderId = selectedFiles.every(
    (file) => (file.folder_id ?? null) === firstFolderId,
  )
    ? firstFolderId
    : null;

  return {
    isMoveOpen,
    setIsMoveOpen,
    isDeleting,
    handleDelete,
    moveSubject: count === 1 ? `“${selectedFiles[0].name}”` : `${count} files`,
    sharedFolderId,
  };
}
