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
  PlugSocketIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { IntegrationMention, SkillCommand } from "../helpers";
import type { MentionOption, MentionTrigger } from "../useChatMentions";

import { Text } from "@/components/atoms/Text/Text";

interface Props {
  /** "/" lists skills; "@" lists integrations, folders and files. */
  trigger?: MentionTrigger;
  /** Skills, or integrations then folders and files — indices match the
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
  trigger = "@",
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
  const resultCount = options.filter(
    (option) => option.kind !== "connect",
  ).length;
  const showEmpty = !isLoading && !isError && resultCount === 0;
  const integrationCount = options.filter(
    (option) => option.kind === "integration",
  ).length;
  const workspaceCount = resultCount - integrationCount;
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
          "flex w-full items-center gap-2 rounded-xl px-3 py-2 text-left text-body",
          isHighlighted
            ? "bg-surface-hover text-foreground-strong"
            : "text-foreground-body",
        )}
      >
        {children}
      </button>
    );
  }

  if (trigger === "/") {
    return (
      <div
        role="listbox"
        aria-label="Skill commands"
        onMouseDown={(e) => e.preventDefault()}
        className="absolute bottom-full left-0 z-50 mb-2 max-h-80 w-80 max-w-full overflow-y-auto rounded-2xl border border-border-base bg-surface p-1.5 shadow-md"
      >
        {options.map((option, index) =>
          option.kind === "skill"
            ? renderOption(
                option,
                index,
                <SkillOption skill={option.skill} />,
                `/${option.skill.name}: ${option.skill.description}`,
              )
            : null,
        )}
        {isLoading ? (
          <p className="flex items-center gap-2 px-3 py-2 text-body text-foreground-tertiary">
            <Icon icon={Loading03Icon} className="h-4 w-4 shrink-0" />
            Loading skills…
          </p>
        ) : options.length === 0 ? (
          <p className="px-3 py-2 text-body text-foreground-tertiary">
            No matching skills.
          </p>
        ) : null}
      </div>
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
      className="absolute bottom-full left-0 z-50 mb-2 flex max-h-96 w-80 max-w-full flex-col overflow-hidden rounded-2xl border border-border-base bg-surface shadow-md"
    >
      <div className="min-h-0 flex-1 overflow-y-auto p-1.5">
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
            return renderOption(
              option,
              index,
              <FileOption file={option.file} />,
            );
          }
          return null;
        })}
        {isError ? (
          <p className="flex items-center gap-2 px-3 py-2 text-body text-error-foreground">
            <Icon icon={AlertCircleIcon} className="h-4 w-4 shrink-0" />
            Couldn&apos;t load files. Try again.
          </p>
        ) : isLoading ? (
          <p className="flex items-center gap-2 px-3 py-2 text-body text-foreground-tertiary">
            <Icon icon={Loading03Icon} spin className="h-4 w-4 shrink-0" />
            Searching files…
          </p>
        ) : null}
        {showEmpty && (
          <p className="px-3 py-2 text-body text-foreground-tertiary">
            {emptyMessage(showFiles, hasIntegrations)}
          </p>
        )}
      </div>
      {/* Pinned below the scrolling results, so connecting another
          integration is always one step away however long the list. */}
      {options.map((option, index) =>
        option.kind === "connect" ? (
          <div key="connect" className="border-t border-border-base p-1.5">
            {renderOption(
              option,
              index,
              <ConnectOption query={option.query} />,
            )}
          </div>
        ) : null,
      )}
    </div>
  );
}

function ConnectOption({ query }: { query: string }) {
  const search = query.trim();
  return (
    <>
      <Icon icon={PlugSocketIcon} tone="primary" className="h-4 w-4 shrink-0" />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-medium">
          Connect an integration
        </span>
        {search && (
          <span className="block truncate text-caption text-foreground-tertiary">
            Search services for “{search}”
          </span>
        )}
      </span>
    </>
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
        <Text
          variant="small-tight"
          as="span"
          tone="muted"
          className="block truncate"
        >
          {integration.providerName}
          {integration.username ? ` · ${integration.username}` : ""}
        </Text>
      </span>
    </>
  );
}

function SkillOption({ skill }: { skill: SkillCommand }) {
  return (
    <span className="min-w-0 flex-1">
      <span className="flex min-w-0 items-baseline gap-2">
        <span className="truncate font-medium">/{skill.name}</span>
        {skill.argumentHint && (
          <span className="shrink-0 text-caption text-foreground-subtle">
            {skill.argumentHint}
          </span>
        )}
      </span>
      <span
        className="block truncate text-caption text-foreground-tertiary"
        title={skill.description}
      >
        {skill.description}
      </span>
    </span>
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
      <Icon icon={Folder01Icon} tone="primary" className="h-4 w-4 shrink-0" />
      <span className="min-w-0 flex-1 truncate">{folder.name}</span>
      <Text variant="small-tight" as="span" tone="muted" className="shrink-0">
        {count}
      </Text>
    </>
  );
}

function FileOption({ file }: { file: WorkspaceFileItem }) {
  return (
    <>
      <Icon
        icon={getFileTypeIcon(file.mime_type)}
        tone="primary"
        className="h-4 w-4 shrink-0"
      />
      <span className="min-w-0 flex-1 truncate">{file.name}</span>
    </>
  );
}

function SectionHeading({ label }: { label: string }) {
  return (
    <p
      role="presentation"
      className="px-3 pt-2 pb-1 text-left text-caption font-medium tracking-wide text-foreground-subtle uppercase"
    >
      {label}
    </p>
  );
}

function mentionOptionKey(option: MentionOption): string {
  if (option.kind === "file") return `file:${option.file.id}`;
  if (option.kind === "folder") return `folder:${option.folder.id}`;
  if (option.kind === "skill") return `skill:${option.skill.name}`;
  if (option.kind === "connect") return "connect";
  return `integration:${option.integration.credentialId}`;
}

function emptyMessage(showFiles: boolean, hasIntegrations: boolean): string {
  if (showFiles && hasIntegrations) return "No matching files or integrations.";
  if (showFiles) return "No matching files.";
  return hasIntegrations
    ? "No matching integrations."
    : "No connected integrations to mention.";
}
