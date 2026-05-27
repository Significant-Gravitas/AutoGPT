"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getListWorkspaceFilesQueryKey,
  useDeleteWorkspaceFile,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import { Button } from "@/components/atoms/Button/Button";
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

  const { mutateAsync: deleteFile, isPending: isDeleting } = useDeleteWorkspaceFile({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getListWorkspaceFilesQueryKey({
            session_id: sessionId ?? undefined,
          }),
        });
        toast({ title: "File deleted", variant: "success" });
      },
      onError: () => toast({ title: "Failed to delete file", variant: "destructive" }),
    },
  });

  function handleOpen(file: SessionFile) {
    if (file.messageID) {
      const el = document.querySelector(`[data-message-id="${file.messageID}"]`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }
    }
    openArtifact(fileItemToArtifactRef(file.item));
  }

  function handleDownload(file: SessionFile) {
    downloadArtifact(fileItemToArtifactRef(file.item)).catch(() =>
      toast({ title: "Download failed", variant: "destructive" }),
    );
  }

  async function handleConfirmDelete() {
    if (!pendingDelete) return;
    await deleteFile({ fileId: pendingDelete.item.id });
    setPendingDelete(null);
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
      <p className="p-6 text-center text-sm text-red-500">
        Failed to load files.
      </p>
    );
  }

  if (isEmpty || (uploaded.length === 0 && generated.length === 0)) {
    return (
      <p className="p-6 text-center text-sm text-zinc-400">
        No files yet. Upload files or ask Otto to create something.
      </p>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center justify-end border-b border-zinc-100 px-3 py-2">
        <Button
          variant="ghost"
          size="small"
          onClick={handleDownloadAll}
          loading={isZipping}
          leftIcon={<DownloadSimpleIcon size={16} />}
        >
          Download all
        </Button>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-2">
        {uploaded.length > 0 && (
          <FileSection title="Uploaded">
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
          <FileSection title="Generated">
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

function FileSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3 className="px-2 pb-1 text-xs font-medium uppercase tracking-wide text-zinc-400">
        {title}
      </h3>
      <div className="flex flex-col">{children}</div>
    </section>
  );
}
