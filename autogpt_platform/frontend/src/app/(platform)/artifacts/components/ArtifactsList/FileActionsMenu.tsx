"use client";

import { useDeleteWorkspaceFile } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { cn } from "@/lib/utils";
import { useQueryClient } from "@tanstack/react-query";
import {
  Delete02Icon,
  Download04Icon,
  Folder01Icon,
  LinkSquare01Icon,
  Loading03Icon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { useState } from "react";
import { ARTIFACTS_LIST_QUERY_KEY } from "../../useArtifactsPage";
import { MoveToFolderDialog } from "../MoveToFolderDialog/MoveToFolderDialog";
import { deriveFileOrigin, downloadFileBlob } from "./helpers";

interface Props {
  file: WorkspaceFileItem;
  className?: string;
}

export function FileActionsMenu({ file, className }: Props) {
  const origin = deriveFileOrigin(file.path);
  const goLabel = origin.kind === "session" ? "Open chat" : "Open in Builder";
  const [isMoveOpen, setIsMoveOpen] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { mutateAsync: deleteFile, isPending: isDeleting } =
    useDeleteWorkspaceFile({
      mutation: {
        onSuccess: () => {
          queryClient.invalidateQueries({
            queryKey: ARTIFACTS_LIST_QUERY_KEY,
          });
          toast({ title: "File deleted" });
        },
        onError: (error) => {
          toast({
            title: "Failed to delete file",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  async function handleDownload() {
    if (isDownloading) return;
    setIsDownloading(true);
    try {
      await downloadFileBlob(file.id, file.name);
    } catch (error) {
      toast({
        title: "Failed to download file",
        description:
          error instanceof Error ? error.message : "Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsDownloading(false);
    }
  }

  async function handleDelete() {
    if (isDeleting) return;
    const confirmed = window.confirm(`Delete "${file.name}"?`);
    if (!confirmed) return;
    await deleteFile({ fileId: file.id });
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            aria-label={`Actions for ${file.name}`}
            className={cn(
              "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900",
              className,
            )}
            data-testid="artifacts-card-menu"
          >
            <Icon icon={MoreHorizontalIcon} size={20} />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-44">
          <DropdownMenuItem
            onSelect={(e) => {
              e.preventDefault();
              handleDownload();
            }}
            disabled={isDownloading}
            data-testid="artifacts-download"
          >
            {isDownloading ? (
              <Icon
                icon={Loading03Icon}
                size={16}
                className="mr-2 animate-spin"
              />
            ) : (
              <Icon icon={Download04Icon} size={16} className="mr-2" />
            )}
            {isDownloading ? "Downloading…" : "Download"}
          </DropdownMenuItem>
          <DropdownMenuItem asChild>
            <Link href={origin.href} data-testid="artifacts-origin-link">
              <Icon icon={LinkSquare01Icon} size={16} className="mr-2" />
              {goLabel}
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem
            onSelect={(e) => {
              e.preventDefault();
              setIsMoveOpen(true);
            }}
            data-testid="artifacts-move-to-folder"
          >
            <Icon icon={Folder01Icon} size={16} className="mr-2" />
            Move to folder
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onSelect={(e) => {
              e.preventDefault();
              handleDelete();
            }}
            disabled={isDeleting}
            className="text-red-600 focus:bg-red-50 focus:text-red-700"
            data-testid="artifacts-delete"
          >
            {isDeleting ? (
              <Icon
                icon={Loading03Icon}
                size={16}
                className="mr-2 animate-spin"
              />
            ) : (
              <Icon icon={Delete02Icon} size={16} className="mr-2" />
            )}
            {isDeleting ? "Deleting…" : "Delete"}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {isMoveOpen && (
        <MoveToFolderDialog
          fileId={file.id}
          fileName={file.name}
          currentFolderId={file.folder_id}
          isOpen={isMoveOpen}
          setIsOpen={setIsMoveOpen}
        />
      )}
    </>
  );
}
