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
      onSuccess: () => {
        invalidate();
        toast({ title: "File moved" });
      },
      onError: () => {
        toast({ title: "Failed to move file", variant: "destructive" });
      },
    },
  });

  return {
    folders: foldersQuery.data?.folders ?? [],
    isLoading: foldersQuery.isLoading,
    createFolder: (args: { name: string }) =>
      createMutation.mutate({ data: { name: args.name } }),
    isCreating: createMutation.isPending,
    updateFolder: (args: { folderId: string; name?: string }) =>
      updateMutation.mutate({
        folderId: args.folderId,
        data: { name: args.name },
      }),
    isUpdating: updateMutation.isPending,
    deleteFolder: (folderId: string) => deleteMutation.mutate({ folderId }),
    isDeleting: deleteMutation.isPending,
    moveFileToFolder: (args: { fileId: string; folderId: string | null }) =>
      moveMutation.mutate({
        data: { file_ids: [args.fileId], folder_id: args.folderId },
      }),
  };
}
