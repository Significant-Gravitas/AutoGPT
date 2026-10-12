"use client";

import { FileViewerModal } from "@/app/(platform)/artifacts/components/FileViewerModal/FileViewerModal";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Attachment01Icon, Download04Icon } from "@hugeicons/core-free-icons";
import { useCardFilesPopover } from "./useCardFilesPopover";

interface Props {
  sessionId: string;
  fileCount: number;
}

export function CardFilesPopover({ sessionId, fileCount }: Props) {
  const {
    open,
    setOpen,
    viewing,
    closeViewer,
    files,
    isLoading,
    isError,
    isEmpty,
    handleOpenFile,
    handleDownload,
  } = useCardFilesPopover(sessionId);

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
          {isLoading && (
            <div className="flex flex-col gap-1.5 p-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-2/3" />
            </div>
          )}
          {isError && (
            <Text variant="small" tone="danger" className="px-2 py-1.5">
              Couldn&apos;t load the files.
            </Text>
          )}
          {isEmpty && (
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
          onClose={closeViewer}
        />
      )}
    </>
  );
}
