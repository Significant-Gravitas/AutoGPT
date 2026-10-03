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
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import type { SelectionModifiers } from "./helpers";
import {
  type KeyboardEvent,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  files: WorkspaceFileItem[];
  selectedIds: ReadonlyMap<string, WorkspaceFileItem>;
  onSelect: (index: number, modifiers: SelectionModifiers) => void;
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  hasMore: boolean;
  isLoadingMore: boolean;
  onLoadMore: () => void;
  emptyMessage: string;
  /** Offered under the empty message, e.g. widening an expert-only listing. */
  emptyAction?: ReactNode;
}

export function WorkspaceFileList({
  files,
  selectedIds,
  onSelect,
  isLoading,
  isError,
  error,
  hasMore,
  isLoadingMore,
  onLoadMore,
  emptyMessage,
  emptyAction,
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
      <div className="flex flex-col items-center gap-2 py-8">
        <p className="text-center text-sm text-zinc-500">{emptyMessage}</p>
        {emptyAction}
      </div>
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
          {files.map((file, index) => {
            const isSelected = selectedIds.has(file.id);
            const fileIcon = getFileTypeIcon(file.mime_type);
            return (
              <button
                key={file.id}
                type="button"
                onClick={(e) =>
                  onSelect(index, {
                    shift: e.shiftKey,
                    meta: e.metaKey || e.ctrlKey,
                  })
                }
                onKeyDown={(e) => handleRangeKey(e, index, onSelect)}
                aria-pressed={isSelected}
                className={cn(
                  "flex w-full items-center gap-3 rounded-2xl border bg-white p-3 text-left transition-colors",
                  isSelected
                    ? "border-violet-300 ring-1 ring-violet-200"
                    : "border-zinc-200 hover:border-zinc-300",
                )}
              >
                <Icon
                  icon={fileIcon}
                  size={20}
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
                  <Icon
                    icon={CheckmarkCircle02Icon}
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

/**
 * Firefox reports shiftKey as false on the click it synthesises from
 * Shift+Enter/Space (Chromium reports true), so the keyboard range cannot ride
 * that click. Handle the chord here and preventDefault, which also stops
 * Chromium firing a second, plain selection.
 */
function handleRangeKey(
  e: KeyboardEvent<HTMLButtonElement>,
  index: number,
  onSelect: (index: number, modifiers: SelectionModifiers) => void,
) {
  if (!e.shiftKey || !isKey(e, "Enter", " ")) return;
  e.preventDefault();
  onSelect(index, { shift: true });
}
