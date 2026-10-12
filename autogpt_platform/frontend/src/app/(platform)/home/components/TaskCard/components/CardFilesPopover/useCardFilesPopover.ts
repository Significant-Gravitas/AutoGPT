import { downloadArtifact } from "@/app/(platform)/copilot/components/ArtifactPanel/downloadArtifact";
import { fileItemToArtifactRef } from "@/app/(platform)/copilot/components/ContextPanel/components/FilesTab/helpers";
import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";

export function useCardFilesPopover(sessionId: string) {
  const [open, setOpen] = useState(false);
  const [viewing, setViewing] = useState<WorkspaceFileItem | null>(null);
  const { toast } = useToast();
  const filesQuery = useListWorkspaceFiles(
    { session_id: sessionId },
    {
      query: {
        enabled: open,
        select: (res) => (res.status === 200 ? res.data.files : []),
      },
    },
  );

  function handleOpenFile(file: WorkspaceFileItem) {
    setOpen(false);
    setViewing(file);
  }

  function handleDownload(file: WorkspaceFileItem) {
    downloadArtifact(fileItemToArtifactRef(file)).catch(() =>
      toast({ title: "Download failed", variant: "destructive" }),
    );
  }

  return {
    open,
    setOpen,
    viewing,
    closeViewer: () => setViewing(null),
    files: filesQuery.data ?? [],
    isLoading: filesQuery.isLoading,
    isError: filesQuery.isError,
    isEmpty: filesQuery.isSuccess && (filesQuery.data ?? []).length === 0,
    handleOpenFile,
    handleDownload,
  };
}
