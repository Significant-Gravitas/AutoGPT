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
import { ApiError } from "@/lib/autogpt-server-api/helpers";
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

  // Its own instance of the same endpoint: a move reports differently from a
  // rename, and the 409/400 the destination can raise have no rename analogue.
  const moveFolderMutation = useUpdateWorkspaceFolder({
    mutation: {
      onSuccess: () => {
        invalidate();
        toast({ title: "Folder moved" });
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
    createFolder: (args: { name: string; parentId?: string | null }) =>
      createMutation.mutateAsync({
        data: { name: args.name, parent_id: args.parentId ?? null },
      }),
    isCreating: createMutation.isPending,
    updateFolder: (args: { folderId: string; name?: string }) =>
      updateMutation.mutateAsync({
        folderId: args.folderId,
        data: { name: args.name },
      }),
    isUpdating: updateMutation.isPending,
    // `parent_id` is sent explicitly on every move, `null` meaning the root;
    // the backend reads `model_fields_set`, so omitting it would mean "stay".
    moveFolder: (args: {
      folderId: string;
      parentId: string | null;
      name: string;
    }) =>
      moveFolderMutation
        .mutateAsync({
          folderId: args.folderId,
          data: { parent_id: args.parentId },
        })
        .catch((error: unknown) => {
          // The name the 409 message needs is not in the request body —
          // sending it would rename the folder — so the toast lives here.
          toast({
            title: describeFolderMoveError(error, args.name),
            variant: "destructive",
          });
          throw error;
        }),
    isMovingFolder: moveFolderMutation.isPending,
    deleteFolder: (folderId: string) =>
      deleteMutation.mutateAsync({ folderId }),
    isDeleting: deleteMutation.isPending,
    moveFilesToFolder: (args: { fileIds: string[]; folderId: string | null }) =>
      moveMutation.mutateAsync({
        data: { file_ids: args.fileIds, folder_id: args.folderId },
      }),
  };
}

export function describeFolderMoveError(error: unknown, name: string): string {
  const status = error instanceof ApiError ? error.status : null;
  if (status === 409) return `A folder named “${name}” is already there`;
  if (status === 400) return "A folder can't be moved into itself";
  return "Failed to move folder";
}
