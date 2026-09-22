"use client";

import { getFileTypeIcon } from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { cn } from "@/lib/utils";
import type { MutableRefObject, ReactNode } from "react";
import { AlertCircleIcon, Loading03Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { IntegrationMention } from "../helpers";
import type { MentionItem } from "../useChatMentions";

interface Props {
  /** Integrations first, then files — indices match the highlight cursor. */
  items: MentionItem[];
  /** Whether workspace files are part of this picker; drives the file
   *  loading/error/empty copy. */
  showFiles: boolean;
  /** Whether the user has any integration to mention at all. */
  hasIntegrations: boolean;
  isLoading: boolean;
  isError: boolean;
  highlightedIndex: number;
  highlightedRef: MutableRefObject<HTMLButtonElement | null>;
  onSelect: (item: MentionItem) => void;
  onHighlight: (index: number) => void;
}

export function MentionDropdown({
  items,
  showFiles,
  hasIntegrations,
  isLoading,
  isError,
  highlightedIndex,
  highlightedRef,
  onSelect,
  onHighlight,
}: Props) {
  const showEmpty = !isLoading && !isError && items.length === 0;
  const integrationCount = items.filter(
    (item) => item.kind === "integration",
  ).length;
  const fileCount = items.length - integrationCount;
  // Headings only earn their space when both kinds can appear together.
  const showHeadings = showFiles && integrationCount > 0;
  const showFilesHeading =
    showHeadings && (fileCount > 0 || isLoading || isError);

  function renderOption(item: MentionItem, index: number, children: ReactNode) {
    const isHighlighted = index === highlightedIndex;
    return (
      <button
        key={mentionItemKey(item)}
        ref={isHighlighted ? highlightedRef : undefined}
        type="button"
        role="option"
        aria-selected={isHighlighted}
        // preventDefault on mousedown keeps focus in the textarea so the
        // caret/selection used to strip the @query stays valid.
        onMouseDown={(e) => {
          e.preventDefault();
          onSelect(item);
        }}
        onMouseEnter={() => onHighlight(index)}
        className={cn(
          "flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-sm",
          isHighlighted ? "bg-zinc-100 text-zinc-900" : "text-zinc-700",
        )}
      >
        {children}
      </button>
    );
  }

  return (
    <div
      role="listbox"
      aria-label="Mention suggestions"
      // preventDefault on mousedown keeps focus in the textarea when clicking
      // non-interactive areas (padding, empty/loading/error states) so the
      // textarea's onBlur doesn't close the dropdown before a selection.
      onMouseDown={(e) => e.preventDefault()}
      className="absolute bottom-full left-0 z-50 mb-2 max-h-60 w-72 overflow-y-auto rounded-2xl border border-zinc-200 bg-white p-1.5 shadow-md"
    >
      {showHeadings && <SectionHeading label="Integrations" />}
      {items.map((item, index) =>
        item.kind === "integration"
          ? renderOption(
              item,
              index,
              <IntegrationOption integration={item.integration} />,
            )
          : null,
      )}
      {showFilesHeading && <SectionHeading label="Files" />}
      {items.map((item, index) =>
        item.kind === "file"
          ? renderOption(item, index, <FileOption file={item.file} />)
          : null,
      )}
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
        <p className="px-3 py-2 text-sm text-zinc-500">
          {emptyMessage(showFiles, hasIntegrations)}
        </p>
      )}
    </div>
  );
}

function IntegrationOption({
  integration,
}: {
  integration: IntegrationMention;
}) {
  return (
    <>
      <IntegrationLogo
        provider={integration.provider}
        alt=""
        size={16}
        className="shrink-0"
      />
      <span className="min-w-0 flex-1 truncate">{integration.name}</span>
      <span className="shrink-0 font-mono text-xs text-zinc-400">
        {integration.token}
      </span>
    </>
  );
}

function FileOption({ file }: { file: WorkspaceFileItem }) {
  return (
    <>
      <Icon
        icon={getFileTypeIcon(file.mime_type)}
        className="h-4 w-4 shrink-0 text-zinc-900"
      />
      <span className="min-w-0 flex-1 truncate">{file.name}</span>
    </>
  );
}

function SectionHeading({ label }: { label: string }) {
  return (
    <p
      role="presentation"
      className="px-3 pb-1 pt-2 text-xs font-medium uppercase tracking-wide text-zinc-400"
    >
      {label}
    </p>
  );
}

function mentionItemKey(item: MentionItem): string {
  return item.kind === "file"
    ? `file:${item.file.id}`
    : `integration:${item.integration.provider}`;
}

function emptyMessage(showFiles: boolean, hasIntegrations: boolean): string {
  if (showFiles && hasIntegrations) return "No matching files or integrations.";
  if (showFiles) return "No matching files.";
  return hasIntegrations
    ? "No matching integrations."
    : "No connected integrations to mention.";
}
