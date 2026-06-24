"use client";

import { useDeleteWorkspaceFile } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { ARTIFACTS_LIST_QUERY_KEY } from "../../../useArtifactsPage";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  ArrowSquareOutIcon,
  CircleNotchIcon,
  DotsThreeIcon,
  DownloadSimpleIcon,
  TrashIcon,
} from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import { motion, useReducedMotion } from "framer-motion";
import type { Variants } from "framer-motion";
import Link from "next/link";
import { useState } from "react";
import {
  deriveFileOrigin,
  downloadFileBlob,
  formatFileSize,
  formatRelativeDate,
  getFileTypeIcon,
  getFileTypeLabel,
} from "../helpers";
import { CardPreview } from "./CardPreview";

interface Props {
  file: WorkspaceFileItem;
  onOpen: (file: WorkspaceFileItem) => void;
}

const CARD_VARIANTS: Variants = {
  hidden: { opacity: 0, y: 8, scale: 0.98, filter: "blur(8px)" },
  show: {
    opacity: 1,
    y: 0,
    scale: 1,
    filter: "blur(0px)",
    transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
  },
};

const REDUCED_CARD_VARIANTS: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: 0.2 } },
};

export function ArtifactCard({ file, onOpen }: Props) {
  const origin = deriveFileOrigin(file.path);
  const goLabel = origin.kind === "session" ? "Open chat" : "Open in Builder";
  const TypeIcon = getFileTypeIcon(file.mime_type, file.name);
  const reduceMotion = useReducedMotion();

  return (
    <motion.li
      variants={reduceMotion ? REDUCED_CARD_VARIANTS : CARD_VARIANTS}
      style={{ willChange: "transform, opacity, filter" }}
      className="group relative flex flex-col overflow-hidden rounded-2xl border border-zinc-200 bg-white transition-colors hover:border-zinc-300"
      data-testid="artifacts-list-item"
    >
      {/* Full-card click target: opening the file is the primary action.
          Sits behind the content (z-0); the content is pointer-events-none so
          clicks fall through, except the kebab menu which re-enables them. */}
      <button
        type="button"
        onClick={() => onOpen(file)}
        aria-label={`Open ${file.name}`}
        className="absolute inset-0 z-0 cursor-pointer"
        data-testid="artifacts-card-open"
      />
      <div className="pointer-events-none relative z-10">
        <CardPreview file={file} />
        <div className="flex items-center gap-3 p-3">
          <TypeIcon
            size={20}
            weight="regular"
            className="shrink-0 text-zinc-500"
          />
          <div className="flex min-w-0 flex-1 flex-col">
            <Text
              variant="body-medium"
              className="truncate text-zinc-900"
              title={file.name}
            >
              {file.name}
            </Text>
            <Text variant="small" className="truncate text-zinc-500">
              {getFileTypeLabel(file.mime_type, file.name)} ·{" "}
              {formatFileSize(file.size_bytes)} ·{" "}
              {formatRelativeDate(file.created_at)}
            </Text>
          </div>
          <div className="pointer-events-auto">
            <CardMenu file={file} goLabel={goLabel} goHref={origin.href} />
          </div>
        </div>
      </div>
    </motion.li>
  );
}

function CardMenu({
  file,
  goLabel,
  goHref,
}: {
  file: WorkspaceFileItem;
  goLabel: string;
  goHref: string;
}) {
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
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label={`Actions for ${file.name}`}
          className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900"
          data-testid="artifacts-card-menu"
        >
          <DotsThreeIcon size={20} weight="bold" />
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
            <CircleNotchIcon size={16} className="mr-2 animate-spin" />
          ) : (
            <DownloadSimpleIcon size={16} className="mr-2" />
          )}
          {isDownloading ? "Downloading…" : "Download"}
        </DropdownMenuItem>
        <DropdownMenuItem asChild>
          <Link href={goHref} data-testid="artifacts-origin-link">
            <ArrowSquareOutIcon size={16} className="mr-2" />
            {goLabel}
          </Link>
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
            <CircleNotchIcon size={16} className="mr-2 animate-spin" />
          ) : (
            <TrashIcon size={16} className="mr-2" />
          )}
          {isDeleting ? "Deleting…" : "Delete"}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
