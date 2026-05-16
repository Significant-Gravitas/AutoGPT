import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { MagnifyingGlassIcon, XIcon } from "@phosphor-icons/react";
import { useEffect, useRef } from "react";
import type { KeyboardEvent } from "react";
import { useCopilotChatRuntimeStore } from "../../copilotChatRegistry";
import { useCopilotUIStore } from "../../store";
import { ChatSearchResults } from "./ChatSearchResults";
import type { SearchSession } from "./helpers";
import { useChatSearch } from "./useChatSearch";

interface Props {
  sessions: SearchSession[];
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
}

export function ChatSearchModal({
  sessions,
  currentSessionId,
  onSelectSession,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const isSearchOpen = useCopilotUIStore((state) => state.isSearchOpen);
  const setSearchOpen = useCopilotUIStore((state) => state.setSearchOpen);
  const completedSessionIDs = useCopilotUIStore(
    (state) => state.completedSessionIDs,
  );
  const sessionNeedsReload = useCopilotChatRuntimeStore(
    (state) => state.sessionNeedsReload,
  );
  const {
    query,
    debouncedQuery,
    setQuery,
    clearQuery,
    results,
    highlightedIndex,
    setHighlightedIndex,
    highlightedResultRef,
    moveHighlight,
    isSearching,
  } = useChatSearch(sessions, isSearchOpen);

  useEffect(() => {
    if (!isSearchOpen) return;
    window.setTimeout(() => inputRef.current?.focus(), 0);
  }, [isSearchOpen]);

  if (!isSearchOpen) return null;

  function closeSearch() {
    setSearchOpen(false);
  }

  function selectHighlightedSession() {
    const session = results[highlightedIndex];
    if (!session) return;
    onSelectSession(session.id);
    closeSearch();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeSearch();
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      moveHighlight(1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      moveHighlight(-1);
    } else if (event.key === "Enter") {
      event.preventDefault();
      selectHighlightedSession();
    }
  }

  return (
    <div
      className="fixed inset-0 z-[80] flex items-start justify-center bg-black/20 px-4 pt-[18vh] backdrop-blur-sm"
      onMouseDown={closeSearch}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="chat-search-title"
        className="w-full max-w-xl overflow-hidden rounded-3xl bg-white shadow-2xl ring-1 ring-zinc-200"
        onMouseDown={(event) => event.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <div className="flex items-center gap-3 bg-zinc-50 p-3">
          <MagnifyingGlassIcon className="h-5 w-5 shrink-0 text-zinc-800" />
          <Input
            ref={inputRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search chats..."
            aria-label="Search chats"
            aria-controls="chat-search-results"
            aria-activedescendant={
              results[highlightedIndex]
                ? `chat-search-result-${results[highlightedIndex].id}`
                : undefined
            }
            autoComplete="off"
            className="h-9 border-0 bg-transparent px-0 text-base text-zinc-950 placeholder:text-zinc-700 shadow-none focus-visible:ring-0"
          />
          {query ? (
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-label="Clear search"
              onClick={clearQuery}
              className="shrink-0"
            >
              <XIcon className="h-4 w-4" />
            </Button>
          ) : null}
          <Button
            type="button"
            variant="outline"
            size="icon-sm"
            aria-label="Close search"
            onClick={closeSearch}
            className="shrink-0 rounded-full border-zinc-400"
          >
            <XIcon className="h-4 w-4" />
          </Button>
        </div>
        <Separator />
        <div className="py-2">
          <div
            id="chat-search-title"
            className="px-4 pb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-700"
          >
            {isSearching ? "Results" : "Recent chats"}
          </div>
          <div
            id="chat-search-results"
            className={cn(
              "max-h-[22rem] overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200",
              results.length === 0 && "max-h-none",
            )}
          >
            {results.length > 0 ? (
              <div className="px-2">
                <ChatSearchResults
                  results={results}
                  query={debouncedQuery}
                  highlightedIndex={highlightedIndex}
                  highlightedResultRef={highlightedResultRef}
                  currentSessionId={currentSessionId}
                  completedSessionIDs={completedSessionIDs}
                  sessionNeedsReload={sessionNeedsReload}
                  onHighlight={setHighlightedIndex}
                  onSelect={(id) => {
                    onSelectSession(id);
                    closeSearch();
                  }}
                />
              </div>
            ) : (
              <div className="px-3 py-8 text-center text-sm text-zinc-500">
                No chats found
              </div>
            )}
          </div>
        </div>
        <Separator />
        <div className="flex items-center justify-between gap-4 bg-zinc-50 px-4 py-4 text-xs text-zinc-700">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <kbd className="inline-flex h-5 w-5 items-center justify-center rounded-md border border-zinc-200 bg-white font-sans text-[11px] text-zinc-800 shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)]">
                ↑
              </kbd>
              <kbd className="inline-flex h-5 w-5 items-center justify-center rounded-md border border-zinc-200 bg-white font-sans text-[11px] text-zinc-800 shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)]">
                ↓
              </kbd>
              <span>Navigate</span>
            </div>
            <div className="flex items-center gap-1.5">
              <kbd className="inline-flex h-5 min-w-5 items-center justify-center rounded-md border border-zinc-200 bg-white px-1 font-sans text-[11px] text-zinc-800 shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)]">
                ↵
              </kbd>
              <span>Select</span>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <kbd className="inline-flex h-5 items-center justify-center rounded-md border border-zinc-200 bg-white px-1.5 font-sans text-[11px] text-zinc-800 shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)]">
              esc
            </kbd>
            <span>Close</span>
          </div>
        </div>
      </div>
    </div>
  );
}
