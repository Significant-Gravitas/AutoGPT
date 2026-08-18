import { listWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { useKeyboardNav } from "@/components/organisms/SearchCommandModal/useKeyboardNav";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { KeyboardEvent } from "react";
import { useState } from "react";

const MENTION_RE = /(?:^|\s)@([^\s@]*)$/;
const QUERY_DEBOUNCE_MS = 200;
const MENTION_RESULT_LIMIT = 8;

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
}: Args) {
  const [active, setActive] = useState<ActiveMention | null>(null);
  const isOpen = enabled && active !== null;

  const debouncedQuery = useDebouncedValue(
    active?.query ?? "",
    QUERY_DEBOUNCE_MS,
  );

  const search = useQuery({
    queryKey: ["chat-mention", "workspace-files", debouncedQuery],
    queryFn: () =>
      listWorkspaceFiles({
        limit: MENTION_RESULT_LIMIT,
        q: debouncedQuery || undefined,
      }),
    enabled: isOpen,
    placeholderData: keepPreviousData,
  });

  const files =
    search.data?.status === 200 ? (search.data.data.files ?? []) : [];

  const {
    highlightedIndex,
    highlightedRef,
    moveHighlight,
    setHighlightedIndex,
  } = useKeyboardNav(files.length, debouncedQuery);

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

  function accept(item: WorkspaceFileItem | undefined) {
    // The highlighted index is clamped in an effect, so a shrinking result
    // list can momentarily leave it pointing past the end — guard against the
    // out-of-bounds `undefined` before touching the item.
    if (!active || !item) return;
    setValue(value.slice(0, active.start) + value.slice(active.end));
    addWorkspaceFile(item);
    setActive(null);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>): boolean {
    if (!isOpen) return false;
    if (e.key === "Escape") {
      e.preventDefault();
      close();
      return true;
    }
    if (files.length === 0) return false;
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        moveHighlight(1);
        return true;
      case "ArrowUp":
        e.preventDefault();
        moveHighlight(-1);
        return true;
      case "Enter":
      case "Tab":
        e.preventDefault();
        accept(files[highlightedIndex]);
        return true;
      default:
        return false;
    }
  }

  return {
    isOpen,
    files,
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
