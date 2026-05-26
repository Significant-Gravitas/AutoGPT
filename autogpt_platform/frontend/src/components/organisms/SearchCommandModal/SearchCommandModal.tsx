import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { MagnifyingGlassIcon, XIcon } from "@phosphor-icons/react";
import {
  useEffect,
  useId,
  useRef,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import {
  flattenBuckets,
  getTotalCount,
  type SearchCommandBucket,
  type SearchCommandItem,
} from "./helpers";
import { SearchCommandResults } from "./SearchCommandResults";
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
  /** Shown while ``isLoading`` is true and there's nothing to display. */
  loadingLabel?: ReactNode;
  /** Shown when ``isError`` is true. Replaces the result list entirely. */
  errorLabel?: ReactNode;
  isLoading?: boolean;
  isError?: boolean;
  placeholder?: string;
  inputAriaLabel?: string;
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
  loadingLabel = "Searching…",
  errorLabel = "Something went wrong. Try again.",
  isLoading = false,
  isError = false,
  placeholder = "Search…",
  inputAriaLabel = "Search",
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

  useEffect(() => {
    if (!isOpen) return;
    // Defer focus until after the dialog has mounted so the input is in
    // the DOM. ``window.setTimeout(…, 0)`` matches the established
    // pattern across the codebase.
    window.setTimeout(() => inputRef.current?.focus(), 0);
  }, [isOpen]);

  if (!isOpen) return null;

  const highlightedFlat = flatResults[highlightedIndex];

  function selectHighlightedItem() {
    if (!highlightedFlat) return;
    onSelectItem(highlightedFlat.item, highlightedFlat.bucketKey);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      onClose();
    } else if (event.key === "ArrowDown") {
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

  const showResults = !isError && totalCount > 0;
  const showLoading = !isError && totalCount === 0 && isLoading;
  const showEmptyState = !isError && totalCount === 0 && !isLoading;

  return (
    <div
      className="fixed inset-0 z-[80] flex items-start justify-center bg-black/20 px-4 pt-[18vh] backdrop-blur-sm"
      onMouseDown={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${idPrefix}-title`}
        className="w-full max-w-xl overflow-hidden rounded-3xl bg-white shadow-2xl ring-1 ring-zinc-200"
        onMouseDown={(event) => event.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
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
              />
            ) : showLoading ? (
              <div
                aria-label="Loading"
                className="px-3 py-8 text-center text-sm text-zinc-400"
              >
                {loadingLabel}
              </div>
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
      </div>
    </div>
  );
}
