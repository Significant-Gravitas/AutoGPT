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
import { useState } from "react";
import { isKey } from "@/lib/keyboard";

const MENTION_RE = /(?:^|\s)@([^\s@]*)$/;
const QUERY_DEBOUNCE_MS = 200;
const MENTION_RESULT_LIMIT = 8;
const MENTION_FOLDER_LIMIT = 3;

interface ActiveMention {
  query: string;
  start: number;
  end: number;
}

interface Args {
  enabled: boolean;
  value: string;
  setValue: (value: string) => void;
  addWorkspaceFile: (item: WorkspaceFileItem) => void;
  addWorkspaceFolder: (folder: WorkspaceFolder, subfolderCount: number) => void;
  /** Expert the chat is scoped to; suggests only files that expert can attach. */
  expertId?: string | null;
}

/**
 * Detects an active `@token` at the textarea caret and drives a workspace-file
 * autocomplete. Selecting a file strips the `@query` from the message and adds
 * the file as an attachment chip. Keyboard nav stays in the textarea (focus
 * never leaves), so this owns the highlight cursor and the key handler.
 */
export function useChatMentions({
  enabled,
  value,
  setValue,
  addWorkspaceFile,
  addWorkspaceFolder,
  expertId,
}: Args) {
  const [active, setActive] = useState<ActiveMention | null>(null);
  const isOpen = enabled && active !== null;

  const debouncedQuery = useDebouncedValue(
    active?.query ?? "",
    QUERY_DEBOUNCE_MS,
  );

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
    enabled: isOpen,
    // Keep results while the same expert's query refines; drop them when the
    // chat switches expert so no foreign file can be picked mid-request.
    placeholderData: (previousData, previousQuery) =>
      previousQuery?.queryKey[3] === (expertId ?? null)
        ? previousData
        : undefined,
  });

  const files =
    search.data?.status === 200 ? (search.data.data.files ?? []) : [];

  const foldersQuery = useListWorkspaceFolders({
    query: { select: okData, enabled: isOpen },
  });
  const allFolders = foldersQuery.data?.folders ?? [];
  const folders = matchFolders(allFolders, debouncedQuery);
  // Folders come first, so the keyboard cursor spans both groups as one list.
  const options: MentionOption[] = [
    ...folders.map((folder) => ({
      kind: "folder" as const,
      folder,
      subfolderCount: subfolderCountOf(allFolders, folder.id),
    })),
    ...files.map((file) => ({ kind: "file" as const, file })),
  ];

  const {
    highlightedIndex,
    highlightedRef,
    moveHighlight,
    setHighlightedIndex,
  } = useKeyboardNav(options.length, debouncedQuery);

  function detect(textarea: HTMLTextAreaElement) {
    if (!enabled) return;
    const caret = textarea.selectionStart ?? textarea.value.length;
    const match = textarea.value.slice(0, caret).match(MENTION_RE);
    if (!match) {
      setActive(null);
      return;
    }
    const query = match[1];
    setActive({ query, start: caret - query.length - 1, end: caret });
  }

  function close() {
    setActive(null);
  }

  function accept(option: MentionOption | undefined) {
    // The highlighted index is clamped in an effect, so a shrinking result
    // list can momentarily leave it pointing past the end — guard against the
    // out-of-bounds `undefined` before touching the option.
    if (!active || !option) return;
    setValue(value.slice(0, active.start) + value.slice(active.end));
    if (option.kind === "folder")
      addWorkspaceFolder(option.folder, option.subfolderCount);
    else addWorkspaceFile(option.file);
    setActive(null);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>): boolean {
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
    options,
    isLoading: search.isLoading,
    isError: search.isError,
    highlightedIndex,
    highlightedRef,
    setHighlightedIndex,
    detect,
    close,
    accept,
    onKeyDown,
  };
}

export type MentionOption =
  | { kind: "file"; file: WorkspaceFileItem }
  | { kind: "folder"; folder: WorkspaceFolder; subfolderCount: number };

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
