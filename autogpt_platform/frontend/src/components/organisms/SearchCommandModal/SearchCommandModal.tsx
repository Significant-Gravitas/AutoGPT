import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { MagnifyingGlassIcon, XIcon } from "@phosphor-icons/react";
import * as RXDialog from "@radix-ui/react-dialog";
import { useId, useRef, type KeyboardEvent, type ReactNode } from "react";
import {
  flattenBuckets,
  getTotalCount,
  type SearchCommandBucket,
  type SearchCommandItem,
} from "./helpers";
import { SearchCommandResults } from "./SearchCommandResults";
import { SearchCommandSkeleton } from "./SearchCommandSkeleton";
import { useKeyboardNav } from "./useKeyboardNav";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  query: string;
  onQueryChange: (next: string) => void;
  /** Controlled bucketed results. Bucket order is preserved as-is. */
  buckets: SearchCommandBucket[];
  onSelectItem: (item: SearchCommandItem, bucketKey: string) => void;
  /** Shown when results are empty and ``query`` is empty. */
  idleEmptyLabel?: ReactNode;
  /** Shown when results are empty and ``query`` is non-empty. */
  searchingEmptyLabel?: ReactNode;
  /** Shown when ``isError`` is true. Replaces the result list entirely. */
  errorLabel?: ReactNode;
  isLoading?: boolean;
  isError?: boolean;
  placeholder?: string;
  inputAriaLabel?: string;
  /** Id of the row whose action is in-flight (renders a spinner). */
  loadingItemId?: string;
}

export function SearchCommandModal({
  isOpen,
  onClose,
  query,
  onQueryChange,
  buckets,
  onSelectItem,
  idleEmptyLabel = "No items",
  searchingEmptyLabel = "No results found",
  errorLabel = "Something went wrong. Try again.",
  isLoading = false,
  isError = false,
  placeholder = "Search…",
  inputAriaLabel = "Search",
  loadingItemId,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const reactId = useId();
  // Strip the colons React uses in ``useId`` so the result survives a
  // selector / id-based lookup in tests and CSS.
  const idPrefix = `search-cmd-${reactId.replace(/:/g, "")}`;

  const flatResults = flattenBuckets(buckets);
  const totalCount = getTotalCount(buckets);
  const trimmedQuery = query.trim();
  const isSearching = trimmedQuery.length > 0;

  const {
    highlightedIndex,
    setHighlightedIndex,
    highlightedRef,
    moveHighlight,
  } = useKeyboardNav(totalCount, trimmedQuery);

  const highlightedFlat = flatResults[highlightedIndex];

  function selectHighlightedItem() {
    if (!highlightedFlat) return;
    onSelectItem(highlightedFlat.item, highlightedFlat.bucketKey);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    // Escape closure is handled by Radix's ``onOpenChange``; arrow
    // keys and Enter stay here so the input keeps focus while the user
    // navigates the result list.
    if (event.key === "ArrowDown") {
      event.preventDefault();
      moveHighlight(1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      moveHighlight(-1);
    } else if (event.key === "Enter") {
      event.preventDefault();
      selectHighlightedItem();
    }
  }

  function handleOpenAutoFocus(event: Event) {
    // Radix's default is to focus the first focusable element. For
    // search-as-you-type we want the input regardless of where it
    // sits in the tab order.
    event.preventDefault();
    inputRef.current?.focus();
  }

  const showResults = !isError && totalCount > 0;
  const showLoading = !isError && totalCount === 0 && isLoading;
  const showEmptyState = !isError && totalCount === 0 && !isLoading;

  return (
    <RXDialog.Root
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <RXDialog.Portal>
        <RXDialog.Overlay className="fixed inset-0 z-[80] bg-black/20 backdrop-blur-sm" />
        <RXDialog.Content
          onOpenAutoFocus={handleOpenAutoFocus}
          className="fixed left-1/2 top-[18vh] z-[80] w-[calc(100%-2rem)] max-w-xl -translate-x-1/2 overflow-hidden rounded-3xl bg-white shadow-2xl ring-1 ring-zinc-200 focus:outline-none"
          onKeyDown={handleKeyDown}
        >
          <RXDialog.Title className="sr-only">Search</RXDialog.Title>
          <div className="flex items-center gap-3 bg-zinc-50 p-3">
            <MagnifyingGlassIcon className="h-5 w-5 shrink-0 text-zinc-800" />
            <Input
              ref={inputRef}
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              placeholder={placeholder}
              aria-label={inputAriaLabel}
              aria-controls={`${idPrefix}-results`}
              aria-activedescendant={
                highlightedFlat
                  ? `${idPrefix}-${highlightedFlat.item.id}`
                  : undefined
              }
              autoComplete="off"
              className="h-9 border-0 bg-transparent px-0 text-base text-zinc-950 shadow-none placeholder:text-zinc-700 focus-visible:ring-0"
            />
            {isLoading && isSearching ? (
              <LoadingSpinner
                size="small"
                aria-label="Searching"
                className="shrink-0 text-zinc-500"
              />
            ) : null}
            {query ? (
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                aria-label="Clear search"
                onClick={() => onQueryChange("")}
                className="shrink-0"
              >
                <XIcon className="h-4 w-4" />
              </Button>
            ) : null}
          </div>
          <Separator />
          <div className="py-2">
            <div
              id={`${idPrefix}-results`}
              className={cn(
                "max-h-[26rem] overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200",
                totalCount === 0 && "max-h-none",
              )}
            >
              {isError ? (
                <div className="px-3 py-8 text-center text-sm text-red-500">
                  {errorLabel}
                </div>
              ) : showResults ? (
                <SearchCommandResults
                  buckets={buckets}
                  idPrefix={idPrefix}
                  query={trimmedQuery}
                  highlightedIndex={highlightedIndex}
                  highlightedRef={highlightedRef}
                  onHighlight={setHighlightedIndex}
                  onSelect={onSelectItem}
                  loadingItemId={loadingItemId}
                />
              ) : showLoading ? (
                // Skeleton over text: the input-side spinner already
                // signals "searching", so the body should mirror the
                // shape of what's about to appear (staggered rows) rather
                // than block the eye with a centred string. Emil
                // Kowalski's strategy-feedback-immediate rule.
                <SearchCommandSkeleton />
              ) : showEmptyState ? (
                <div className="px-3 py-8 text-center text-sm text-zinc-500">
                  {isSearching ? searchingEmptyLabel : idleEmptyLabel}
                </div>
              ) : null}
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
        </RXDialog.Content>
      </RXDialog.Portal>
    </RXDialog.Root>
  );
}
