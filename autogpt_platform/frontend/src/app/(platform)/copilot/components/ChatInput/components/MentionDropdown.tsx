"use client";

import { getFileTypeIcon } from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { folderSummary } from "@/app/(platform)/artifacts/components/WorkspaceFolders/folderTree";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { cn } from "@/lib/utils";
import type { MutableRefObject, ReactNode } from "react";
import {
  AlertCircleIcon,
  Folder01Icon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { IntegrationMention } from "../helpers";
import type { MentionOption } from "../useChatMentions";

interface Props {
  /** Integrations first, then folders and files — indices match the
   *  highlight cursor. */
  options: MentionOption[];
  /** Whether workspace files are part of this picker; drives the file
   *  loading/error/empty copy. */
  showFiles: boolean;
  /** Whether the user has any integration to mention at all. */
  hasIntegrations: boolean;
  isLoading: boolean;
  isError: boolean;
  highlightedIndex: number;
  highlightedRef: MutableRefObject<HTMLButtonElement | null>;
  onSelect: (option: MentionOption) => void;
  onHighlight: (index: number) => void;
}

export function MentionDropdown({
  options,
  showFiles,
  hasIntegrations,
  isLoading,
  isError,
  highlightedIndex,
  highlightedRef,
  onSelect,
  onHighlight,
}: Props) {
  const showEmpty = !isLoading && !isError && options.length === 0;
  const integrationCount = options.filter(
    (option) => option.kind === "integration",
  ).length;
  const workspaceCount = options.length - integrationCount;
  // Headings only earn their space when both kinds can appear together.
  const showHeadings = showFiles && integrationCount > 0;
  const showFilesHeading =
    showHeadings && (workspaceCount > 0 || isLoading || isError);

  function renderOption(
    option: MentionOption,
    index: number,
    children: ReactNode,
    ariaLabel?: string,
  ) {
    const isHighlighted = index === highlightedIndex;
    return (
      <button
        key={mentionOptionKey(option)}
        ref={isHighlighted ? highlightedRef : undefined}
        type="button"
        role="option"
        aria-selected={isHighlighted}
        aria-label={ariaLabel}
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
      className="absolute bottom-full left-0 z-50 mb-2 max-h-80 w-80 max-w-full overflow-y-auto rounded-2xl border border-zinc-200 bg-white p-1.5 shadow-md"
    >
      {showHeadings && <SectionHeading label="Integrations" />}
      {options.map((option, index) =>
        option.kind === "integration"
          ? renderOption(
              option,
              index,
              <IntegrationOption integration={option.integration} />,
            )
          : null,
      )}
      {showFilesHeading && <SectionHeading label="Files" />}
      {options.map((option, index) => {
        if (option.kind === "folder") {
          const count = folderSummary(
            option.folder.file_count ?? 0,
            option.subfolderCount,
          );
          return renderOption(
            option,
            index,
            <FolderOption folder={option.folder} count={count} />,
            `Folder ${option.folder.name}, ${count}`,
          );
        }
        if (option.kind === "file") {
          return renderOption(option, index, <FileOption file={option.file} />);
        }
        return null;
      })}
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
        provider={
          integration.provider === "codex" ? "openai" : integration.provider
        }
        alt=""
        size={16}
        className="shrink-0"
      />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-medium" title={integration.name}>
          {integration.name}
        </span>
        <span className="block truncate text-xs text-zinc-500">
          {integration.providerName}
          {integration.username ? ` · ${integration.username}` : ""}
        </span>
      </span>
    </>
  );
}

function FolderOption({
  folder,
  count,
}: {
  folder: WorkspaceFolder;
  count: string;
}) {
  return (
    <>
      <Icon icon={Folder01Icon} className="h-4 w-4 shrink-0 text-zinc-900" />
      <span className="min-w-0 flex-1 truncate">{folder.name}</span>
      <span className="shrink-0 text-xs text-zinc-500">{count}</span>
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
      className="px-3 pb-1 pt-2 text-left text-xs font-medium uppercase tracking-wide text-zinc-400"
    >
      {label}
    </p>
  );
}

function mentionOptionKey(option: MentionOption): string {
  if (option.kind === "file") return `file:${option.file.id}`;
  if (option.kind === "folder") return `folder:${option.folder.id}`;
  return `integration:${option.integration.credentialId}`;
}

function emptyMessage(showFiles: boolean, hasIntegrations: boolean): string {
  if (showFiles && hasIntegrations) return "No matching files or integrations.";
  if (showFiles) return "No matching files.";
  return hasIntegrations
    ? "No matching integrations."
    : "No connected integrations to mention.";
}
