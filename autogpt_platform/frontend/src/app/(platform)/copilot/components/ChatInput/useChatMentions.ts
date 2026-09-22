import { listWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
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
const QUERY_DEBOUNCE_MS = 200;
const MENTION_RESULT_LIMIT = 8;
const INTEGRATION_RESULT_LIMIT = 6;

interface ActiveMention {
  query: string;
  start: number;
  end: number;
}

export type MentionItem =
  | { kind: "file"; file: WorkspaceFileItem }
  | { kind: "integration"; integration: IntegrationMention };

interface Args {
  enabled: boolean;
  value: string;
  setValue: (value: string) => void;
  addWorkspaceFile: (item: WorkspaceFileItem) => void;
  /** Expert the chat is scoped to; suggests only files that expert can attach. */
  expertId?: string | null;
  /** Off when the workspace-files flag is: the picker then only offers
   *  integrations and never queries the file API. */
  includeWorkspaceFiles?: boolean;
  /** Connected integrations offered above the file results. */
  integrations?: IntegrationMention[];
}

/**
 * Detects an active `@token` at the textarea caret and drives a mention
 * autocomplete over connected integrations and workspace files. Selecting a
 * file strips the `@query` from the message and adds the file as an
 * attachment chip; selecting an integration replaces the `@query` with the
 * integration's `@Token` so the reference stays in the prompt text. Keyboard
 * nav stays in the textarea (focus never leaves), so this owns the highlight
 * cursor and the key handler.
 */
export function useChatMentions({
  enabled,
  value,
  setValue,
  addWorkspaceFile,
  expertId,
  includeWorkspaceFiles = true,
  integrations = [],
}: Args) {
  const [active, setActive] = useState<ActiveMention | null>(null);
  const isOpen = enabled && active !== null;
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
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
      }),
    enabled: isOpen && includeWorkspaceFiles,
    // Keep results while the same expert's query refines; drop them when the
    // chat switches expert so no foreign file can be picked mid-request.
    placeholderData: (previousData, previousQuery) =>
      previousQuery?.queryKey[3] === (expertId ?? null)
        ? previousData
        : undefined,
  });

  const files =
    includeWorkspaceFiles && search.data?.status === 200
      ? (search.data.data.files ?? [])
      : [];
  const matchedIntegrations = filterIntegrationMentions(
    integrations,
    query,
  ).slice(0, INTEGRATION_RESULT_LIMIT);

  const items: MentionItem[] = [
    ...matchedIntegrations.map(
      (integration): MentionItem => ({ kind: "integration", integration }),
    ),
    ...files.map((file): MentionItem => ({ kind: "file", file })),
  ];

  const {
    highlightedIndex,
    highlightedRef,
    moveHighlight,
    setHighlightedIndex,
  } = useKeyboardNav(items.length, query);

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

  function detect(textarea: HTMLTextAreaElement) {
    if (!enabled) return;
    textareaRef.current = textarea;
    const caret = textarea.selectionStart ?? textarea.value.length;
    const match = textarea.value.slice(0, caret).match(MENTION_RE);
    if (!match) {
      setActive(null);
      return;
    }
    const token = match[1];
    setActive({ query: token, start: caret - token.length - 1, end: caret });
  }

  function close() {
    setActive(null);
  }

  function accept(item: MentionItem | undefined) {
    // The highlighted index is clamped in an effect, so a shrinking result
    // list can momentarily leave it pointing past the end — guard against the
    // out-of-bounds `undefined` before touching the item.
    if (!active || !item) return;
    if (item.kind === "file") {
      setValue(value.slice(0, active.start) + value.slice(active.end));
      addWorkspaceFile(item.file);
    } else {
      const next = insertIntegrationMention(value, active, item.integration);
      setValue(next.value);
      setPendingCaret(next.caret);
    }
    setActive(null);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>): boolean {
    if (!isOpen) return false;
    if (isKey(e, "Escape")) {
      e.preventDefault();
      close();
      return true;
    }
    if (items.length === 0) return false;
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
      accept(items[highlightedIndex]);
      return true;
    }
    return false;
  }

  return {
    isOpen,
    items,
    showFiles: includeWorkspaceFiles,
    hasIntegrations: integrations.length > 0,
    isLoading: includeWorkspaceFiles && search.isLoading,
    isError: includeWorkspaceFiles && search.isError,
    highlightedIndex,
    highlightedRef,
    setHighlightedIndex,
    detect,
    close,
    accept,
    onKeyDown,
  };
}
