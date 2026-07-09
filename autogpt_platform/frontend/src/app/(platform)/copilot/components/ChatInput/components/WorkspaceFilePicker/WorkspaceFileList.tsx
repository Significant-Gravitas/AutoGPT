"use client";

import {
  formatFileSize,
  formatRelativeDate,
  getFileTypeIcon,
  getFileTypeLabel,
} from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";
import { CheckCircle as CheckCircleIcon } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";

interface Props {
  files: WorkspaceFileItem[];
  selectedIds: ReadonlyMap<string, WorkspaceFileItem>;
  onToggle: (item: WorkspaceFileItem) => void;
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  hasMore: boolean;
  isLoadingMore: boolean;
  onLoadMore: () => void;
}

export function WorkspaceFileList({
  files,
  selectedIds,
  onToggle,
  isLoading,
  isError,
  error,
  hasMore,
  isLoadingMore,
  onLoadMore,
}: Props) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  // Show the top/bottom scroll-fade only when there's content hidden in that
  // direction — no fade when pinned to the very top or bottom.
  const [edges, setEdges] = useState({ top: false, bottom: false });

  function updateEdges() {
    const el = scrollRef.current;
    if (!el) return;
    setEdges({
      top: el.scrollTop > 1,
      bottom: el.scrollTop + el.clientHeight < el.scrollHeight - 1,
    });
  }

  useEffect(() => {
    updateEdges();
  }, [files.length, isLoadingMore]);

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 gap-2 py-2">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-16 w-full rounded-2xl" />
        ))}
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorCard
        responseError={{
          message: error instanceof Error ? error.message : "Failed to load",
        }}
        context="workspace files"
      />
    );
  }

  if (files.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-zinc-500">
        No files in your workspace yet.
      </p>
    );
  }

  return (
    <div className="relative">
      {/* White scroll-fade so rows dissolve into the dialog — only on the
          side that still has hidden content. */}
      {edges.top && (
        <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-6 bg-gradient-to-b from-white to-transparent" />
      )}
      {edges.bottom && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 h-6 bg-gradient-to-t from-white to-transparent" />
      )}
      <div
        ref={scrollRef}
        onScroll={updateEdges}
        className="max-h-[24rem] overflow-y-auto py-1"
      >
        <div className="grid grid-cols-2 gap-2">
          {files.map((file) => {
            const isSelected = selectedIds.has(file.id);
            const Icon = getFileTypeIcon(file.mime_type);
            return (
              <button
                key={file.id}
                type="button"
                onClick={() => onToggle(file)}
                aria-pressed={isSelected}
                className={cn(
                  "flex w-full items-center gap-3 rounded-2xl border bg-white p-3 text-left transition-colors",
                  isSelected
                    ? "border-violet-300 ring-1 ring-violet-200"
                    : "border-zinc-200 hover:border-zinc-300",
                )}
              >
                <Icon
                  size={20}
                  weight="bold"
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
                    {getFileTypeLabel(file.mime_type)} ·{" "}
                    {formatFileSize(file.size_bytes)} ·{" "}
                    {formatRelativeDate(file.created_at)}
                  </Text>
                </div>
                {isSelected && (
                  <CheckCircleIcon
                    weight="fill"
                    className="h-5 w-5 shrink-0 text-violet-600"
                  />
                )}
              </button>
            );
          })}
        </div>
        {hasMore && (
          <div className="mt-2 flex justify-center">
            <Button
              type="button"
              variant="ghost"
              size="small"
              onClick={onLoadMore}
              loading={isLoadingMore}
            >
              Load more
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
