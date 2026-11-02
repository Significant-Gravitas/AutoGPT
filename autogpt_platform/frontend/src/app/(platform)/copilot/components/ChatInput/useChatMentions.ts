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
  insertIntegrationMention,
  type IntegrationMention,
} from "./helpers";

const MENTION_RE = /(?:^|\s)@([^\s@]*)$/;
// Account names can contain spaces ("Work Gmail"), so this one keeps matching
// past whitespace; it only wins when the wider token still names an account.
const ACCOUNT_MENTION_RE = /(?:^|\s)@([^@\n]*)$/;
const QUERY_DEBOUNCE_MS = 200;
const MENTION_RESULT_LIMIT = 8;
const MENTION_FOLDER_LIMIT = 3;
const INTEGRATION_RESULT_LIMIT = 6;

export interface MentionInput {
  value: string;
  selectionStart: number | null;
  setSelectionRange: (start: number, end: number) => void;
  focus?: () => void;
}

/** The `@query` under the caret and where it sits in the message. */
export interface ActiveMention {
  query: string;
  start: number;
  end: number;
}

export type MentionOption =
  | { kind: "file"; file: WorkspaceFileItem }
  | { kind: "folder"; folder: WorkspaceFolder; subfolderCount: number }
  | { kind: "integration"; integration: IntegrationMention }
  | { kind: "connect"; query: string };

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
 * `@query` with a credential reference rendered as an inline badge. Keyboard nav stays in the textarea (focus never leaves), so
 * this owns the highlight cursor and the key handler.
 *
 * The list always ends with "Connect an integration". Choosing it sets
 * `connectRequest`, which the host turns into the connect dialog; once an
 * account is connected, `resumeAfterConnect` reopens the same `@query` so
 * the new account can be picked without retyping it.
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
  const isOpen = enabled && active !== null;
  const textareaRef = useRef<MentionInput | null>(null);
  const [pendingCaret, setPendingCaret] = useState<number | null>(null);
  const [connectRequest, setConnectRequest] = useState<ActiveMention | null>(
    null,
  );
  // The request whose connect finishes last may not be the latest one: in an
  // expert chat the grant lands after the dialog closed, and by then the user
  // may have opened Connect again.
  const latestConnectRef = useRef<ActiveMention | null>(null);

  const query = active?.query ?? "";
  const debouncedQuery = useDebouncedValue(query, QUERY_DEBOUNCE_MS);

  const expertKey = expertId ?? null;
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
    enabled: isOpen && includeWorkspaceFiles,
    // Keep results while the same expert's query refines; drop them when the
    // chat switches expert so no foreign file can be picked mid-request.
    placeholderData: (previousData, previousQuery) =>
      previousQuery?.queryKey[3] === expertKey ? previousData : undefined,
  });

  const files =
    includeWorkspaceFiles && search.data?.status === 200
      ? (search.data.data.files ?? [])
      : [];

  const foldersQuery = useListWorkspaceFolders({
    query: { select: okData, enabled: isOpen && includeWorkspaceFiles },
  });
  const allFolders = includeWorkspaceFiles
    ? (foldersQuery.data?.folders ?? [])
    : [];
  const folders = matchFolders(allFolders, debouncedQuery);

  // The connect dialog's search also matches service descriptions, so the
  // account just connected may not match the `@query` it was found by
  // (`@errors` finding Sentry); it is listed first for that mention anyway.
  const [justConnectedId, setJustConnectedId] = useState<string | null>(null);
  const justConnected = integrations.find(
    (integration) => integration.credentialId === justConnectedId,
  );
  const matched = filterIntegrationMentions(integrations, query);
  const matchedIntegrations = (
    justConnected && !matched.includes(justConnected)
      ? [justConnected, ...matched]
      : matched
  ).slice(0, INTEGRATION_RESULT_LIMIT);

  // Integrations, then folders, then files, then the way to connect another
  // integration: the keyboard cursor spans them all as one list.
  const options: MentionOption[] = [
    ...matchedIntegrations.map(
      (integration): MentionOption => ({ kind: "integration", integration }),
    ),
    ...folders.map(
      (folder): MentionOption => ({
        kind: "folder",
        folder,
        subfolderCount: subfolderCountOf(allFolders, folder.id),
      }),
    ),
    ...files.map((file): MentionOption => ({ kind: "file", file })),
    { kind: "connect", query },
  ];

  const {
    highlightedIndex: cursorIndex,
    highlightedRef,
    moveHighlight,
    setHighlightedIndex,
  } = useKeyboardNav(options.length, query);
  // With nothing matched only "Connect an integration" is left, and it waits
  // to be arrowed or pointed to: pre-selecting it would turn Enter after any
  // unmatched `@word` into opening the dialog instead of sending.
  const [pointedAtQuery, setPointedAtQuery] = useState<string | null>(null);
  const hasResults = options.length > 1;
  const highlightedIndex =
    hasResults || pointedAtQuery === query ? cursorIndex : -1;

  function highlight(index: number) {
    setPointedAtQuery(query);
    setHighlightedIndex(index);
  }

  // The caret lands wherever the browser puts it after a programmatic value
  // change (usually the end); move it to just after the inserted mention once
  // the new text has been committed to the textarea.
  useEffect(() => {
    if (pendingCaret === null) return;
    const textarea = textareaRef.current;
    if (textarea && textarea.value.length >= pendingCaret) {
      // A no-op while typing; the connect dialog has no trigger to hand focus
      // back to, so returning from it relies on this.
      textarea.focus?.();
      textarea.setSelectionRange(pendingCaret, pendingCaret);
    }
    setPendingCaret(null);
  }, [pendingCaret, value]);

  // Choosing Connect lasts only for the mention it was made in; a later one
  // with the same text starts with nothing highlighted again.
  function endMention() {
    setActive(null);
    setPointedAtQuery(null);
    setJustConnectedId(null);
  }

  function detect(textarea: MentionInput) {
    if (!enabled) return;
    textareaRef.current = textarea;
    const caret = textarea.selectionStart ?? textarea.value.length;
    const beforeCaret = textarea.value.slice(0, caret);
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
      endMention();
      return;
    }
    const token = match[1];
    setActive({ query: token, start: caret - token.length - 1, end: caret });
  }

  function close() {
    endMention();
  }

  function accept(option: MentionOption | undefined) {
    // The highlighted index is clamped in an effect, so a shrinking result
    // list can momentarily leave it pointing past the end — guard against the
    // out-of-bounds `undefined` before touching the option.
    if (!active || !option) return;
    if (option.kind === "connect") {
      latestConnectRef.current = active;
      setConnectRequest(active);
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
    endMention();
  }

  // Puts the caret back after the `@query` the dialog was opened from, but
  // only while the message is as it was then. In an expert chat the dialog
  // closes before the grant lands, and anything typed meanwhile (`@sentry`
  // becoming `@sentry2`) means the saved range no longer fits.
  function returnToMention(request: ActiveMention): boolean {
    const text = textareaRef.current?.value ?? value;
    if (text !== value) return false;
    if (text.slice(request.start, request.end) !== `@${request.query}`)
      return false;
    setPendingCaret(request.end);
    return true;
  }

  function cancelConnect() {
    if (connectRequest) returnToMention(connectRequest);
    setConnectRequest(null);
  }

  function resumeAfterConnect(request: ActiveMention, credentialId: string) {
    if (latestConnectRef.current !== request) return;
    latestConnectRef.current = null;
    setConnectRequest(null);
    if (!returnToMention(request)) return;
    setActive(request);
    setJustConnectedId(credentialId);
  }

  function onKeyDown(e: KeyboardEvent<HTMLElement>): boolean {
    if (!isOpen) return false;
    if (isKey(e, "Escape")) {
      e.preventDefault();
      close();
      return true;
    }
    if (isKey(e, "ArrowDown")) {
      e.preventDefault();
      setPointedAtQuery(query);
      moveHighlight(1);
      return true;
    }
    if (isKey(e, "ArrowUp")) {
      e.preventDefault();
      setPointedAtQuery(query);
      moveHighlight(-1);
      return true;
    }
    if (highlightedIndex < 0) return false;
    if (isKey(e, "Enter", "Tab")) {
      e.preventDefault();
      accept(options[highlightedIndex]);
      return true;
    }
    return false;
  }

  return {
    isOpen,
    options,
    showFiles: includeWorkspaceFiles,
    hasIntegrations: integrations.length > 0,
    isLoading: includeWorkspaceFiles && search.isLoading,
    isError: includeWorkspaceFiles && search.isError,
    highlightedIndex,
    highlightedRef,
    setHighlightedIndex: highlight,
    bindInput: (input: MentionInput) => {
      textareaRef.current = input;
    },
    detect,
    close,
    accept,
    onKeyDown,
    connectRequest,
    cancelConnect,
    resumeAfterConnect,
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
