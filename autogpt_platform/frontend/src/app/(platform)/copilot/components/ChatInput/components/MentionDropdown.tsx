"use client";

import { getFileTypeIcon } from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { cn } from "@/lib/utils";
import type { MutableRefObject } from "react";
import { AlertCircleIcon, Loading03Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  files: WorkspaceFileItem[];
  isLoading: boolean;
  isError: boolean;
  highlightedIndex: number;
  highlightedRef: MutableRefObject<HTMLButtonElement | null>;
  onSelect: (item: WorkspaceFileItem) => void;
  onHighlight: (index: number) => void;
}

export function MentionDropdown({
  files,
  isLoading,
  isError,
  highlightedIndex,
  highlightedRef,
  onSelect,
  onHighlight,
}: Props) {
  const showEmpty = !isLoading && !isError && files.length === 0;

  return (
    <div
      role="listbox"
      aria-label="Workspace files"
      // preventDefault on mousedown keeps focus in the textarea when clicking
      // non-interactive areas (padding, empty/loading/error states) so the
      // textarea's onBlur doesn't close the dropdown before a selection.
      onMouseDown={(e) => e.preventDefault()}
      className="absolute bottom-full left-0 z-50 mb-2 max-h-60 w-72 overflow-y-auto rounded-2xl border border-zinc-200 bg-white p-1.5 shadow-md"
    >
      {isError ? (
        <p className="flex items-center gap-2 px-3 py-2 text-sm text-red-600">
          <Icon icon={AlertCircleIcon} className="h-4 w-4 shrink-0" />
          Couldn&apos;t load files. Try again.
        </p>
      ) : isLoading ? (
        <p className="flex items-center gap-2 px-3 py-2 text-sm text-zinc-500">
          <Icon
            icon={Loading03Icon}
            className="h-4 w-4 shrink-0 animate-spin"
          />
          Searching files…
        </p>
      ) : null}
      {showEmpty && (
        <p className="px-3 py-2 text-sm text-zinc-500">No matching files.</p>
      )}
      {files.map((file, index) => {
        const isHighlighted = index === highlightedIndex;
        const fileIcon = getFileTypeIcon(file.mime_type);
        return (
          <button
            key={file.id}
            ref={isHighlighted ? highlightedRef : undefined}
            type="button"
            role="option"
            aria-selected={isHighlighted}
            // preventDefault on mousedown keeps focus in the textarea so the
            // caret/selection used to strip the @query stays valid.
            onMouseDown={(e) => {
              e.preventDefault();
              onSelect(file);
            }}
            onMouseEnter={() => onHighlight(index)}
            className={cn(
              "flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm",
              isHighlighted ? "bg-zinc-100 text-zinc-900" : "text-zinc-700",
            )}
          >
            <Icon icon={fileIcon} className="h-4 w-4 shrink-0 text-zinc-900" />
            <span className="min-w-0 flex-1 truncate">{file.name}</span>
          </button>
        );
      })}
    </div>
  );
}
