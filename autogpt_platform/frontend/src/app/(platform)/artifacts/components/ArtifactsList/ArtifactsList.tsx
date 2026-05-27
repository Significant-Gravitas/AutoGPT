"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import {
  ArrowSquareOutIcon,
  CircleNotchIcon,
  DownloadSimpleIcon,
} from "@phosphor-icons/react";
import Link from "next/link";
import { useState } from "react";
import {
  deriveBadgeLabel,
  FileIllustration,
  pickFileTypeKey,
} from "./FileIllustration";
import { FileTypeMarquee } from "./FileTypeMarquee";
import {
  deriveFileOrigin,
  formatFileSize,
  formatRelativeDate,
  getFileDownloadUrl,
} from "./helpers";

interface Props {
  files: WorkspaceFileItem[];
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  hasSearchTerm: boolean;
}

export function ArtifactsList({
  files,
  isLoading,
  isError,
  error,
  hasSearchTerm,
}: Props) {
  if (isLoading) {
    return (
      <div className="flex flex-col gap-2" data-testid="artifacts-loading">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-16 w-full" />
        ))}
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorCard
        context="artifacts"
        responseError={
          error instanceof Error ? { message: error.message } : undefined
        }
      />
    );
  }

  if (files.length === 0) {
    return (
      <div
        className="flex min-h-[20rem] flex-col items-center justify-center gap-4 p-8 text-center"
        data-testid="artifacts-empty"
      >
        <FileTypeMarquee />
        <Text variant="h5" className="text-zinc-700">
          {hasSearchTerm ? "No files match your search" : "No artifacts yet"}
        </Text>
      </div>
    );
  }

  return (
    <ul
      className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3"
      data-testid="artifacts-list"
    >
      {files.map((file) => (
        <li
          key={file.id}
          className="group flex items-center gap-3 rounded-2xl border border-zinc-200 bg-white p-1 px-2 pr-4 transition-colors hover:cursor-pointer hover:border-zinc-300"
          data-testid="artifacts-list-item"
        >
          <div className="inline-flex shrink-0 items-center justify-center rounded-xl bg-zinc-100 p-4 ring-1 ring-zinc-100">
            <FileIllustration
              typeKey={pickFileTypeKey(file.mime_type)}
              label={deriveBadgeLabel(file.name, file.mime_type)}
              size="sm"
              badgeClassName="!bottom-3 -right-6 px-2.5 py-1.5 text-xs"
            />
          </div>
          <div className="flex min-w-0 flex-1 flex-col gap-1 self-stretch py-2">
            <Text
              variant="body-medium"
              className="truncate text-zinc-900"
              title={file.name}
            >
              {file.name}
            </Text>
            <Text variant="small" className="text-zinc-500">
              {formatFileSize(file.size_bytes)} ·{" "}
              {formatRelativeDate(file.created_at)}
            </Text>
            <FileCardActions file={file} />
          </div>
        </li>
      ))}
    </ul>
  );
}

function DownloadButton({ file }: { file: WorkspaceFileItem }) {
  const [isDownloading, setIsDownloading] = useState(false);

  async function handleDownload() {
    if (isDownloading) return;
    setIsDownloading(true);
    try {
      const res = await fetch(getFileDownloadUrl(file.id));
      if (!res.ok) {
        throw new Error(`Download failed: ${res.status}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } finally {
      setIsDownloading(false);
    }
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={handleDownload}
          disabled={isDownloading}
          aria-label={`Download ${file.name}`}
          className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-zinc-200 text-zinc-700 transition-colors hover:bg-zinc-100 hover:text-zinc-900 disabled:cursor-wait disabled:opacity-70 disabled:hover:bg-transparent"
          data-testid="artifacts-download"
        >
          {isDownloading ? (
            <CircleNotchIcon size={16} className="animate-spin" />
          ) : (
            <DownloadSimpleIcon size={16} />
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent>
        {isDownloading ? "Downloading…" : "Download"}
      </TooltipContent>
    </Tooltip>
  );
}

function FileCardActions({ file }: { file: WorkspaceFileItem }) {
  const origin = deriveFileOrigin(file.path);
  const goLabel = origin.kind === "session" ? "Open chat" : "Open in Builder";

  return (
    <TooltipProvider delayDuration={150}>
      <div className="mt-auto flex items-center justify-end gap-2 pt-2">
        <DownloadButton file={file} />
        <Tooltip>
          <TooltipTrigger asChild>
            <Link
              href={origin.href}
              aria-label={goLabel}
              className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-zinc-200 text-zinc-700 transition-colors hover:bg-zinc-100 hover:text-zinc-900"
              data-testid="artifacts-origin-link"
            >
              <ArrowSquareOutIcon size={16} />
            </Link>
          </TooltipTrigger>
          <TooltipContent>{goLabel}</TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
}
