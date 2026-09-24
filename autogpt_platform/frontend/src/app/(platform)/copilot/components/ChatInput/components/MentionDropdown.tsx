"use client";

import { getFileTypeIcon } from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { folderSummary } from "@/app/(platform)/artifacts/components/WorkspaceFolders/folderTree";
import { cn } from "@/lib/utils";
import type { MutableRefObject } from "react";
import {
  AlertCircleIcon,
  Folder01Icon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";
import type { MentionOption } from "../useChatMentions";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  options: MentionOption[];
  isLoading: boolean;
  isError: boolean;
  highlightedIndex: number;
  highlightedRef: MutableRefObject<HTMLButtonElement | null>;
  onSelect: (option: MentionOption) => void;
  onHighlight: (index: number) => void;
}

export function MentionDropdown({
  options,
  isLoading,
  isError,
  highlightedIndex,
  highlightedRef,
  onSelect,
  onHighlight,
}: Props) {
  const showEmpty = !isLoading && !isError && options.length === 0;

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
      {options.map((option, index) => {
        const isHighlighted = index === highlightedIndex;
        const isFolder = option.kind === "folder";
        const name = isFolder ? option.folder.name : option.file.name;
        const count = isFolder
          ? folderSummary(option.folder.file_count ?? 0, option.subfolderCount)
          : null;
        return (
          <button
            key={isFolder ? `folder-${option.folder.id}` : option.file.id}
            ref={isHighlighted ? highlightedRef : undefined}
            type="button"
            role="option"
            aria-selected={isHighlighted}
            aria-label={isFolder ? `Folder ${name}, ${count}` : undefined}
            // preventDefault on mousedown keeps focus in the textarea so the
            // caret/selection used to strip the @query stays valid.
            onMouseDown={(e) => {
              e.preventDefault();
              onSelect(option);
            }}
            onMouseEnter={() => onHighlight(index)}
            className={cn(
              "flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm",
              isHighlighted ? "bg-zinc-100 text-zinc-900" : "text-zinc-700",
            )}
          >
            <Icon
              icon={
                isFolder ? Folder01Icon : getFileTypeIcon(option.file.mime_type)
              }
              className="h-4 w-4 shrink-0 text-zinc-900"
            />
            <span className="min-w-0 flex-1 truncate">{name}</span>
            {count ? (
              <span className="shrink-0 text-xs text-zinc-500">{count}</span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
