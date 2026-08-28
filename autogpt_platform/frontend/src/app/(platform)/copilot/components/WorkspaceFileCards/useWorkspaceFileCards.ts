"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getListWorkspaceFilesQueryKey,
  useDeleteWorkspaceFile,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useCopilotUIStore } from "../../store";
import { downloadArtifact } from "../ArtifactPanel/downloadArtifact";
import {
  downloadFilesAsZip,
  fileItemToArtifactRef,
} from "../ContextPanel/components/FilesTab/helpers";
import {
  useSessionFiles,
  type SessionFile,
} from "../ContextPanel/components/FilesTab/useSessionFiles";

export function useWorkspaceFileCards(sessionId: string | null) {
  const queryClient = useQueryClient();
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const tenantScope = useCopilotUIStore((s) => s.artifactTenantScope);
  // Session-scoped on purpose: the card is this chat's drawer — only files
  // uploaded to or created in this session. The artifacts side panel carries
  // the workspace-wide library.
  const { uploaded, generated, isLoading, isError } =
    useSessionFiles(sessionId);
  const [pendingDelete, setPendingDelete] = useState<SessionFile | null>(null);
  const [isZipping, setIsZipping] = useState(false);

  const { mutateAsync: deleteFile, isPending: isDeleting } =
    useDeleteWorkspaceFile({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            predicate: ({ queryKey }) => {
              const listRoot = getListWorkspaceFilesQueryKey()[0];
              const scopedSuffix = getTeamScopedQueryKey(
                [],
                tenantScope?.organizationId,
                tenantScope?.teamId,
              )[0];
              return (
                queryKey[0] === listRoot &&
                JSON.stringify(queryKey[queryKey.length - 1]) ===
                  JSON.stringify(scopedSuffix)
              );
            },
          });
          toast({ title: "File deleted", variant: "success" });
        },
        onError: () =>
          toast({ title: "Failed to delete file", variant: "destructive" }),
      },
      request: getTenantRequestInit(
        tenantScope?.organizationId,
        tenantScope?.teamId,
        tenantScope !== null,
      ),
    });

  const files = [...generated, ...uploaded];

  function handleOpen(file: SessionFile) {
    openArtifact(fileItemToArtifactRef(file.item));
  }

  function handleDownload(file: SessionFile) {
    downloadArtifact(fileItemToArtifactRef(file.item)).catch(() =>
      toast({ title: "Download failed", variant: "destructive" }),
    );
  }

  async function handleConfirmDelete() {
    if (!pendingDelete) return;
    try {
      await deleteFile({ fileId: pendingDelete.item.id });
    } catch {
      // onError already toasted; swallow so the dialog closes either way.
    } finally {
      setPendingDelete(null);
    }
  }

  async function handleDownloadAll() {
    if (files.length === 0) return;
    setIsZipping(true);
    try {
      await downloadFilesAsZip(
        files.map((f) => ({
          id: f.item.id,
          name: f.item.name,
          organizationId: f.item.organization_id ?? null,
          teamId: f.item.team_id ?? null,
        })),
      );
    } catch {
      toast({ title: "Download all failed", variant: "destructive" });
    } finally {
      setIsZipping(false);
    }
  }

  return {
    files,
    isLoading,
    isError,
    isDeleting,
    isZipping,
    pendingDelete,
    setPendingDelete,
    handleOpen,
    handleDownload,
    handleConfirmDelete,
    handleDownloadAll,
  };
}
