"use client";

import { FileViewerModal } from "@/app/(platform)/artifacts/components/FileViewerModal/FileViewerModal";
import { downloadArtifact } from "@/app/(platform)/copilot/components/ArtifactPanel/downloadArtifact";
import { fileItemToArtifactRef } from "@/app/(platform)/copilot/components/ContextPanel/components/FilesTab/helpers";
import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Attachment01Icon, Download04Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";

interface Props {
  sessionId: string;
  fileCount: number;
}

export function CardFilesPopover({ sessionId, fileCount }: Props) {
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
  const files = filesQuery.data ?? [];

  function handleOpenFile(file: WorkspaceFileItem) {
    setOpen(false);
    setViewing(file);
  }

  function handleDownload(file: WorkspaceFileItem) {
    downloadArtifact(fileItemToArtifactRef(file)).catch(() =>
      toast({ title: "Download failed", variant: "destructive" }),
    );
  }

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            aria-label={`${fileCount} ${fileCount === 1 ? "file" : "files"}`}
            onClick={(event) => event.stopPropagation()}
            className="relative z-10 flex items-center gap-0.5 rounded px-1 text-xs tabular-nums text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-800"
          >
            <Icon icon={Attachment01Icon} size={14} />
            {fileCount}
          </button>
        </PopoverTrigger>
        <PopoverContent align="end" className="w-72 p-1.5">
          <Text variant="small-medium" tone="muted" className="px-2 py-1">
            Files
          </Text>
          {filesQuery.isLoading && (
            <div className="flex flex-col gap-1.5 p-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-2/3" />
            </div>
          )}
          {filesQuery.isError && (
            <Text variant="small" tone="danger" className="px-2 py-1.5">
              Couldn&apos;t load the files.
            </Text>
          )}
          {filesQuery.isSuccess && files.length === 0 && (
            <Text variant="small" tone="muted" className="px-2 py-1.5">
              No files yet.
            </Text>
          )}
          <ul className="flex max-h-64 flex-col overflow-y-auto">
            {files.map((file) => (
              <li
                key={file.id}
                className="flex items-center gap-1 rounded-md px-2 py-1 hover:bg-zinc-50"
              >
                <Text variant="small" className="min-w-0 flex-1 truncate">
                  {file.name}
                </Text>
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() => handleOpenFile(file)}
                >
                  Open
                </Button>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label={`Download ${file.name}`}
                  onClick={() => handleDownload(file)}
                >
                  <Icon icon={Download04Icon} size={14} />
                </Button>
              </li>
            ))}
          </ul>
        </PopoverContent>
      </Popover>
      {viewing && (
        <FileViewerModal
          key={viewing.id}
          file={viewing}
          onClose={() => setViewing(null)}
        />
      )}
    </>
  );
}
