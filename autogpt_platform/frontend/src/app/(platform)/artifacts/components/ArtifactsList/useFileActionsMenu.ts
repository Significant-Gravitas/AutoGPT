import {
  getGetWorkspaceStorageUsageQueryKey,
  getListWorkspaceFoldersQueryKey,
  useDeleteWorkspaceFile,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ARTIFACTS_LIST_QUERY_KEY } from "../../useArtifactsPage";
import { deriveFileOrigin, downloadFileBlob } from "./helpers";

export function useFileActionsMenu(file: WorkspaceFileItem) {
  const origin = deriveFileOrigin(file.path);
  const [isMoveOpen, setIsMoveOpen] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { mutateAsync: deleteFile, isPending: isDeleting } =
    useDeleteWorkspaceFile({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: ARTIFACTS_LIST_QUERY_KEY,
          });
          // Folder rows show file counts and the footer shows storage used,
          // so both go stale after a deletion.
          queryClient.invalidateQueries({
            queryKey: getListWorkspaceFoldersQueryKey(),
          });
          queryClient.invalidateQueries({
            queryKey: getGetWorkspaceStorageUsageQueryKey(),
          });
          toast({ title: "File deleted" });
        },
        onError: (error) => {
          toast({
            title: "Failed to delete file",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  async function handleDownload() {
    if (isDownloading) return;
    setIsDownloading(true);
    try {
      await downloadFileBlob(file.id, file.name);
    } catch (error) {
      toast({
        title: "Failed to download file",
        description:
          error instanceof Error ? error.message : "Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsDownloading(false);
    }
  }

  async function handleDelete() {
    if (isDeleting) return;
    const confirmed = window.confirm(`Delete "${file.name}"?`);
    if (!confirmed) return;
    try {
      await deleteFile({ fileId: file.id });
    } catch {
      // Already surfaced by the mutation's onError toast.
    }
  }

  return {
    goLabel: origin.kind === "session" ? "Open chat" : "Open in Builder",
    goHref: origin.href,
    isMoveOpen,
    setIsMoveOpen,
    isDownloading,
    isDeleting,
    handleDownload,
    handleDelete,
  };
}
