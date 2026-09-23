import { useQueryClient } from "@tanstack/react-query";
import {
  getListWorkspaceFoldersQueryKey,
  useBulkMoveWorkspaceFiles,
  useCreateWorkspaceFolder,
  useDeleteWorkspaceFolder,
  useListWorkspaceFolders,
  useUpdateWorkspaceFolder,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ARTIFACTS_LIST_QUERY_KEY } from "./useArtifactsPage";

export function useArtifactsFolders() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const foldersQuery = useListWorkspaceFolders({
    query: { select: okData },
  });

  function invalidate() {
    queryClient.invalidateQueries({
      queryKey: getListWorkspaceFoldersQueryKey(),
    });
    queryClient.invalidateQueries({ queryKey: ARTIFACTS_LIST_QUERY_KEY });
  }

  const createMutation = useCreateWorkspaceFolder({
    mutation: {
      onSuccess: () => {
        invalidate();
        toast({ title: "Folder created" });
      },
      onError: () => {
        toast({
          title: "Failed to create folder",
          description: "A folder with this name may already exist.",
          variant: "destructive",
        });
      },
    },
  });

  const updateMutation = useUpdateWorkspaceFolder({
    mutation: {
      onSuccess: () => {
        invalidate();
        toast({ title: "Folder updated" });
      },
      onError: () => {
        toast({
          title: "Failed to update folder",
          variant: "destructive",
        });
      },
    },
  });

  const deleteMutation = useDeleteWorkspaceFolder({
    mutation: {
      onSuccess: () => {
        invalidate();
        toast({ title: "Folder deleted", description: "Files moved to root." });
      },
      onError: () => {
        toast({
          title: "Failed to delete folder",
          variant: "destructive",
        });
      },
    },
  });

  const moveMutation = useBulkMoveWorkspaceFiles({
    mutation: {
      onSuccess: (_, variables) => {
        invalidate();
        const count = variables.data.file_ids.length;
        toast({ title: count === 1 ? "File moved" : `${count} files moved` });
      },
      onError: (_, variables) => {
        const count = variables.data.file_ids.length;
        toast({
          title: count === 1 ? "Failed to move file" : "Failed to move files",
          variant: "destructive",
        });
      },
    },
  });

  return {
    folders: foldersQuery.data?.folders ?? [],
    isLoading: foldersQuery.isLoading,
    isError: foldersQuery.isError,
    error: foldersQuery.error,
    createFolder: (args: { name: string }) =>
      createMutation.mutateAsync({ data: { name: args.name } }),
    isCreating: createMutation.isPending,
    updateFolder: (args: { folderId: string; name?: string }) =>
      updateMutation.mutateAsync({
        folderId: args.folderId,
        data: { name: args.name },
      }),
    isUpdating: updateMutation.isPending,
    deleteFolder: (folderId: string) =>
      deleteMutation.mutateAsync({ folderId }),
    isDeleting: deleteMutation.isPending,
    moveFilesToFolder: (args: { fileIds: string[]; folderId: string | null }) =>
      moveMutation.mutateAsync({
        data: { file_ids: args.fileIds, folder_id: args.folderId },
      }),
  };
}
