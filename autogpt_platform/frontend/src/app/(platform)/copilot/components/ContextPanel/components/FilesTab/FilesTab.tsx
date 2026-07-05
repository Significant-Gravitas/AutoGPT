"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getListWorkspaceFilesQueryKey,
  useDeleteWorkspaceFile,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { toast } from "@/components/molecules/Toast/use-toast";
import { Skeleton } from "@/components/ui/skeleton";
import { DownloadSimpleIcon } from "@phosphor-icons/react";
import { useCopilotUIStore } from "../../../../store";
import { downloadArtifact } from "../../../ArtifactPanel/downloadArtifact";
import { DeleteFileDialog } from "./components/DeleteFileDialog";
import { FileRow } from "./components/FileRow";
import { downloadFilesAsZip, fileItemToArtifactRef } from "./helpers";
import { useSessionFiles, type SessionFile } from "./useSessionFiles";

interface Props {
  sessionId: string | null;
}

export function FilesTab({ sessionId }: Props) {
  const queryClient = useQueryClient();
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const { uploaded, generated, isLoading, isError, isEmpty } =
    useSessionFiles(sessionId);
  const [pendingDelete, setPendingDelete] = useState<SessionFile | null>(null);
  const [isZipping, setIsZipping] = useState(false);

  const { mutateAsync: deleteFile, isPending: isDeleting } =
    useDeleteWorkspaceFile({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: getListWorkspaceFilesQueryKey({
              session_id: sessionId ?? undefined,
            }),
          });
          toast({ title: "File deleted", variant: "success" });
        },
        onError: () =>
          toast({ title: "Failed to delete file", variant: "destructive" }),
      },
    });

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
      // onError toast already surfaces the failure; swallow so the await
      // settles and the dialog closes instead of staying open on rejection.
    } finally {
      setPendingDelete(null);
    }
  }

  async function handleDownloadAll() {
    const all = [...uploaded, ...generated].map((f) => ({
      id: f.item.id,
      name: f.item.name,
    }));
    if (all.length === 0) return;
    setIsZipping(true);
    try {
      await downloadFilesAsZip(all);
    } catch {
      toast({ title: "Download all failed", variant: "destructive" });
    } finally {
      setIsZipping(false);
    }
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2 p-3">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="p-3">
        <ErrorCard
          isSuccess={false}
          context="files"
          responseError={{ message: "Failed to load files." }}
        />
      </div>
    );
  }

  if (isEmpty || (uploaded.length === 0 && generated.length === 0)) {
    return (
      <div className="flex h-full flex-1 items-center justify-center p-6">
        <p className="text-center text-sm text-zinc-400">
          No files yet. Upload files or ask Autopilot to create something.
        </p>
      </div>
    );
  }

  const downloadAllButton = (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          onClick={handleDownloadAll}
          loading={isZipping}
          aria-label="Download all"
          className="h-7 !min-w-0 !p-1"
        >
          <DownloadSimpleIcon size={16} />
        </Button>
      </TooltipTrigger>
      <TooltipContent>Download all</TooltipContent>
    </Tooltip>
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-6 py-3">
        {uploaded.length > 0 && (
          <FileSection
            title="Uploaded files"
            action={generated.length === 0 ? downloadAllButton : null}
          >
            {uploaded.map((file) => (
              <FileRow
                key={file.item.id}
                file={file}
                onOpen={handleOpen}
                onDownload={handleDownload}
                onRequestDelete={setPendingDelete}
              />
            ))}
          </FileSection>
        )}
        {generated.length > 0 && (
          <FileSection title="Generated files" action={downloadAllButton}>
            {generated.map((file) => (
              <FileRow
                key={file.item.id}
                file={file}
                onOpen={handleOpen}
                onDownload={handleDownload}
                onRequestDelete={setPendingDelete}
              />
            ))}
          </FileSection>
        )}
      </div>
      <DeleteFileDialog
        fileName={pendingDelete?.item.name ?? null}
        isDeleting={isDeleting}
        onConfirm={handleConfirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}

function FileSection({
  title,
  action,
  children,
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div className="flex items-center justify-between pb-1">
        <h3 className="text-sm font-medium text-zinc-900">{title}</h3>
        {action}
      </div>
      <div className="flex flex-col">{children}</div>
    </section>
  );
}
