import {
  listWorkspaceFiles,
  useListWorkspaceFolders,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { okData } from "@/app/api/helpers";
import { subfolderCountOf } from "@/app/(platform)/artifacts/components/WorkspaceFolders/folderTree";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { useKeyboardNav } from "@/components/organisms/SearchCommandModal/useKeyboardNav";
import { useQuery } from "@tanstack/react-query";
import type { KeyboardEvent } from "react";
import { useEffect, useRef, useState } from "react";
import { isKey } from "@/lib/keyboard";
import {
  filterIntegrationMentions,
  filterSkillCommands,
  insertIntegrationMention,
  insertSkillCommand,
  type IntegrationMention,
  type SkillCommand,
} from "./helpers";
import { useSkillCommands } from "./useSkillCommands";

const MENTION_RE = /(?:^|\s)@([^\s@]*)$/;
// Account names can contain spaces ("Work Gmail"), so this one keeps matching
// past whitespace; it only wins when the wider token still names an account.
const ACCOUNT_MENTION_RE = /(?:^|\s)@([^@\n]*)$/;
// A command is only a leading "/" and a name, so a path ("/etc/hosts") or a
// "/" later in the message never opens the picker.
const COMMAND_RE = /^\/([A-Za-z0-9_.:-]*)$/;
const QUERY_DEBOUNCE_MS = 200;
const MENTION_RESULT_LIMIT = 8;
const MENTION_FOLDER_LIMIT = 3;
const INTEGRATION_RESULT_LIMIT = 6;
const SKILL_RESULT_LIMIT = 8;

export interface MentionInput {
  value: string;
  selectionStart: number | null;
  setSelectionRange: (start: number, end: number) => void;
}

/** `@` mentions an integration or attaches a file; `/` runs a skill. */
export type MentionTrigger = "@" | "/";

interface ActiveMention {
  trigger: MentionTrigger;
  query: string;
  start: number;
  end: number;
}

export type MentionOption =
  | { kind: "file"; file: WorkspaceFileItem }
  | { kind: "folder"; folder: WorkspaceFolder; subfolderCount: number }
  | { kind: "integration"; integration: IntegrationMention }
  | { kind: "skill"; skill: SkillCommand };

interface Args {
  enabled: boolean;
  value: string;
  setValue: (value: string) => void;
  addWorkspaceFile: (item: WorkspaceFileItem) => void;
  addWorkspaceFolder: (folder: WorkspaceFolder, subfolderCount: number) => void;
  /** Expert the chat is scoped to; suggests only files that expert can attach. */
  expertId?: string | null;
  /** False while the workspace-files flag is off: the picker then only
   *  offers integrations and never queries the file or folder APIs. */
  includeWorkspaceFiles?: boolean;
  /** Connected integrations offered above the folder and file results. */
  integrations?: IntegrationMention[];
}

/**
 * Detects an active `@token` at the textarea caret and drives a mention
 * autocomplete over connected integrations, workspace folders and workspace
 * files. Selecting a file or folder strips the `@query` from the message and
 * adds it as an attachment chip; selecting an integration replaces the
 * `@query` with a credential reference rendered as an inline badge. A
 * leading `/name` lists the chat's skills instead; picking one writes
 * `/name ` so the user types its arguments next. Keyboard nav stays in the
 * textarea (focus never leaves), so this owns the highlight cursor and the
 * key handler.
 */
export function useChatMentions({
  enabled,
  value,
  setValue,
  addWorkspaceFile,
  addWorkspaceFolder,
  expertId,
  includeWorkspaceFiles = true,
  integrations = [],
}: Args) {
  const [active, setActive] = useState<ActiveMention | null>(null);
  // Skills are fetched only once a leading "/" shows the user wants them.
  const [wantsSkills, setWantsSkills] = useState(false);
  const skillCommands = useSkillCommands(expertId, enabled && wantsSkills);
  const trigger: MentionTrigger = active?.trigger ?? "@";
  const isCommand = trigger === "/";
  // In a chat with no skills to run, a leading "/" is just text.
  const hasNoCommands =
    isCommand && !skillCommands.isLoading && skillCommands.skills.length === 0;
  const isOpen = enabled && active !== null && !hasNoCommands;
  const showsWorkspace = includeWorkspaceFiles && !isCommand;
  const textareaRef = useRef<MentionInput | null>(null);
  const [pendingCaret, setPendingCaret] = useState<number | null>(null);

  const query = active?.query ?? "";
  const debouncedQuery = useDebouncedValue(query, QUERY_DEBOUNCE_MS);

  const search = useQuery({
    queryKey: [
      "chat-mention",
      "workspace-files",
      debouncedQuery,
      expertId ?? null,
    ] as const,
    queryFn: () =>
      listWorkspaceFiles({
        limit: MENTION_RESULT_LIMIT,
        q: debouncedQuery || undefined,
        expert_id: expertId ?? undefined,
        // A typed name is a deliberate search, so it always spans everything
        // the chat may attach — the picker's narrower default is for browsing.
        include_user_files: expertId ? true : undefined,
      }),
    enabled: isOpen && showsWorkspace,
    // Keep results while the same expert's query refines; drop them when the
    // chat switches expert so no foreign file can be picked mid-request.
    placeholderData: (previousData, previousQuery) =>
      previousQuery?.queryKey[3] === (expertId ?? null)
        ? previousData
        : undefined,
  });

  const files =
    showsWorkspace && search.data?.status === 200
      ? (search.data.data.files ?? [])
      : [];

  const foldersQuery = useListWorkspaceFolders({
    query: { select: okData, enabled: isOpen && showsWorkspace },
  });
  const allFolders = showsWorkspace ? (foldersQuery.data?.folders ?? []) : [];
  const folders = matchFolders(allFolders, debouncedQuery);

  const matchedIntegrations = filterIntegrationMentions(
    integrations,
    query,
  ).slice(0, INTEGRATION_RESULT_LIMIT);

  // "/" lists skills. "@" lists integrations, then folders, then files. The
  // keyboard cursor spans the groups as one list.
  const options: MentionOption[] = isCommand
    ? filterSkillCommands(skillCommands.skills, query)
        .slice(0, SKILL_RESULT_LIMIT)
        .map((skill): MentionOption => ({ kind: "skill", skill }))
    : [
        ...matchedIntegrations.map(
          (integration): MentionOption => ({
            kind: "integration",
            integration,
          }),
        ),
        ...folders.map(
          (folder): MentionOption => ({
            kind: "folder",
            folder,
            subfolderCount: subfolderCountOf(allFolders, folder.id),
          }),
        ),
        ...files.map((file): MentionOption => ({ kind: "file", file })),
      ];

  const {
    highlightedIndex,
    highlightedRef,
    moveHighlight,
    setHighlightedIndex,
  } = useKeyboardNav(options.length, query);

  // The caret lands wherever the browser puts it after a programmatic value
  // change (usually the end); move it to just after the inserted mention once
  // the new text has been committed to the textarea.
  useEffect(() => {
    if (pendingCaret === null) return;
    const textarea = textareaRef.current;
    if (textarea && textarea.value.length >= pendingCaret) {
      textarea.setSelectionRange(pendingCaret, pendingCaret);
    }
    setPendingCaret(null);
  }, [pendingCaret, value]);

  // With workspace files off and nothing connected there is nothing to pick,
  // so a typed `@` must not open an empty dropdown.
  const hasOptions = includeWorkspaceFiles || integrations.length > 0;

  function detect(textarea: MentionInput) {
    if (!enabled) return;
    textareaRef.current = textarea;
    const caret = textarea.selectionStart ?? textarea.value.length;
    const beforeCaret = textarea.value.slice(0, caret);
    const command = beforeCaret.match(COMMAND_RE);
    if (command) {
      setWantsSkills(true);
      setActive({ trigger: "/", query: command[1], start: 0, end: caret });
      return;
    }
    if (!hasOptions) {
      setActive(null);
      return;
    }
    // A single-word token is always a query; a multi-word one is only kept
    // while it still matches a connected account.
    const accountMatch = beforeCaret.match(ACCOUNT_MENTION_RE);
    const match =
      beforeCaret.match(MENTION_RE) ??
      (accountMatch &&
      filterIntegrationMentions(integrations, accountMatch[1]).length > 0
        ? accountMatch
        : null);
    if (!match) {
      setActive(null);
      return;
    }
    const token = match[1];
    setActive({
      trigger: "@",
      query: token,
      start: caret - token.length - 1,
      end: caret,
    });
  }

  function close() {
    setActive(null);
  }

  function accept(option: MentionOption | undefined) {
    // The highlighted index is clamped in an effect, so a shrinking result
    // list can momentarily leave it pointing past the end — guard against the
    // out-of-bounds `undefined` before touching the option.
    if (!active || !option) return;
    if (option.kind === "skill") {
      const next = insertSkillCommand(value, active, option.skill);
      setValue(next.value);
      setPendingCaret(next.caret);
    } else if (option.kind === "integration") {
      const next = insertIntegrationMention(value, active, option.integration);
      setValue(next.value);
      setPendingCaret(next.caret);
    } else {
      setValue(value.slice(0, active.start) + value.slice(active.end));
      if (option.kind === "folder")
        addWorkspaceFolder(option.folder, option.subfolderCount);
      else addWorkspaceFile(option.file);
    }
    setActive(null);
  }

  function onKeyDown(e: KeyboardEvent<HTMLElement>): boolean {
    if (!isOpen) return false;
    if (isKey(e, "Escape")) {
      e.preventDefault();
      close();
      return true;
    }
    if (options.length === 0) return false;
    if (isKey(e, "ArrowDown")) {
      e.preventDefault();
      moveHighlight(1);
      return true;
    }
    if (isKey(e, "ArrowUp")) {
      e.preventDefault();
      moveHighlight(-1);
      return true;
    }
    if (isKey(e, "Enter", "Tab")) {
      e.preventDefault();
      accept(options[highlightedIndex]);
      return true;
    }
    return false;
  }

  return {
    isOpen,
    trigger,
    options,
    showFiles: showsWorkspace,
    hasIntegrations: integrations.length > 0,
    isLoading: isCommand
      ? skillCommands.isLoading
      : includeWorkspaceFiles && search.isLoading,
    isError: showsWorkspace && search.isError,
    highlightedIndex,
    highlightedRef,
    setHighlightedIndex,
    bindInput: (input: MentionInput) => {
      textareaRef.current = input;
    },
    detect,
    close,
    accept,
    onKeyDown,
  };
}

function matchFolders(
  folders: WorkspaceFolder[],
  query: string,
): WorkspaceFolder[] {
  const needle = query.trim().toLowerCase();
  return folders
    .filter((folder) => folder.name.toLowerCase().includes(needle))
    .sort((a, b) => a.name.localeCompare(b.name))
    .slice(0, MENTION_FOLDER_LIMIT);
}
