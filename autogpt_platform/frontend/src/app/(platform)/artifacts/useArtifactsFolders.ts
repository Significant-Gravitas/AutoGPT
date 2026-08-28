import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  bulkMoveWorkspaceFiles,
  createWorkspaceFolder,
  deleteWorkspaceFolder,
  getListWorkspaceFoldersQueryKey,
  updateWorkspaceFolder,
  useListWorkspaceFolders,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { okData } from "@/app/api/helpers";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useOrgTeamStore } from "@/services/org-team/store";
import { ARTIFACTS_LIST_QUERY_KEY } from "./useArtifactsPage";

interface TenantScope {
  organizationId: string | null;
  teamId: string | null;
}

function folderScope(folder: WorkspaceFolder): TenantScope {
  return {
    organizationId: folder.organization_id ?? null,
    teamId: folder.team_id ?? null,
  };
}

export function useArtifactsFolders(requestedScope?: TenantScope) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const activeTeamID = useOrgTeamStore((s) => s.activeTeamID);
  const isTenantReady = useOrgTeamStore((s) => s.isLoaded);
  const organizationId = requestedScope
    ? requestedScope.organizationId
    : activeOrgID;
  const teamId = requestedScope ? requestedScope.teamId : activeTeamID;

  const foldersQuery = useListWorkspaceFolders({
    query: {
      enabled: isTenantReady,
      queryKey: getTeamScopedQueryKey(
        getListWorkspaceFoldersQueryKey(),
        organizationId,
        teamId,
      ),
      select: okData,
    },
    request: getTenantRequestInit(organizationId, teamId, isTenantReady),
  });

  function invalidate() {
    queryClient.invalidateQueries({
      queryKey: getListWorkspaceFoldersQueryKey(),
    });
    queryClient.invalidateQueries({ queryKey: ARTIFACTS_LIST_QUERY_KEY });
  }

  const createMutation = useMutation({
    mutationFn: ({ name }: { name: string }) =>
      createWorkspaceFolder(
        { name },
        getTenantRequestInit(organizationId, teamId),
      ),
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
  });

  const updateMutation = useMutation({
    mutationFn: ({
      folder,
      name,
    }: {
      folder: WorkspaceFolder;
      name?: string;
    }) =>
      updateWorkspaceFolder(
        folder.id,
        { name },
        getTenantRequestInit(
          folder.organization_id ?? null,
          folder.team_id ?? null,
        ),
      ),
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
  });

  const deleteMutation = useMutation({
    mutationFn: (folder: WorkspaceFolder) =>
      deleteWorkspaceFolder(
        folder.id,
        getTenantRequestInit(
          folder.organization_id ?? null,
          folder.team_id ?? null,
        ),
      ),
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
  });

  const moveMutation = useMutation({
    mutationFn: ({
      fileId,
      folderId,
      sourceScope,
    }: {
      fileId: string;
      folderId: string | null;
      sourceScope: TenantScope;
    }) =>
      bulkMoveWorkspaceFiles(
        { file_ids: [fileId], folder_id: folderId },
        getTenantRequestInit(sourceScope.organizationId, sourceScope.teamId),
      ),
    onSuccess: () => {
      invalidate();
      toast({ title: "File moved" });
    },
    onError: () => {
      toast({ title: "Failed to move file", variant: "destructive" });
    },
  });

  return {
    folders: foldersQuery.data?.folders ?? [],
    isLoading: foldersQuery.isLoading,
    isError: foldersQuery.isError,
    error: foldersQuery.error,
    createFolder: (args: { name: string }) =>
      createMutation.mutateAsync({ name: args.name }),
    isCreating: createMutation.isPending,
    updateFolder: (args: { folder: WorkspaceFolder; name?: string }) =>
      updateMutation.mutateAsync(args),
    isUpdating: updateMutation.isPending,
    deleteFolder: (folder: WorkspaceFolder) =>
      deleteMutation.mutateAsync(folder),
    isDeleting: deleteMutation.isPending,
    moveFileToFolder: (args: {
      fileId: string;
      folder: WorkspaceFolder | null;
      sourceScope: TenantScope;
    }) => {
      if (args.folder) {
        const destinationScope = folderScope(args.folder);
        if (
          destinationScope.organizationId !== args.sourceScope.organizationId ||
          destinationScope.teamId !== args.sourceScope.teamId
        ) {
          toast({
            title: "Choose a folder in the same team",
            variant: "destructive",
          });
          return Promise.reject(new Error("Cross-tenant move is not allowed"));
        }
      }
      return moveMutation.mutateAsync({
        fileId: args.fileId,
        folderId: args.folder?.id ?? null,
        sourceScope: args.sourceScope,
      });
    },
  };
}
