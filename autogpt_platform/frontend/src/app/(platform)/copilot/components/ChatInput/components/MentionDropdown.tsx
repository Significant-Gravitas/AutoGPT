"use client";

import { getFileTypeIcon } from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { cn } from "@/lib/utils";
import {
  CircleNotch as CircleNotchIcon,
  WarningCircle as WarningCircleIcon,
} from "@phosphor-icons/react";
import type { RefObject } from "react";

interface Props {
  files: WorkspaceFileItem[];
  isLoading: boolean;
  isError: boolean;
  highlightedIndex: number;
  highlightedRef: RefObject<HTMLButtonElement | null>;
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
      className="absolute bottom-full left-0 z-50 mb-2 max-h-60 w-72 overflow-y-auto rounded-2xl border border-zinc-200 bg-white p-1.5 shadow-md"
    >
      {isError ? (
        <p className="flex items-center gap-2 px-3 py-2 text-sm text-red-600">
          <WarningCircleIcon className="h-4 w-4 shrink-0" />
          Couldn&apos;t load files. Try again.
        </p>
      ) : isLoading ? (
        <p className="flex items-center gap-2 px-3 py-2 text-sm text-zinc-500">
          <CircleNotchIcon className="h-4 w-4 shrink-0 animate-spin" />
          Searching files…
        </p>
      ) : null}
      {showEmpty && (
        <p className="px-3 py-2 text-sm text-zinc-500">No matching files.</p>
      )}
      {files.map((file, index) => {
        const isHighlighted = index === highlightedIndex;
        const Icon = getFileTypeIcon(file.mime_type);
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
            <Icon weight="regular" className="h-4 w-4 shrink-0 text-zinc-900" />
            <span className="min-w-0 flex-1 truncate">{file.name}</span>
          </button>
        );
      })}
    </div>
  );
}
