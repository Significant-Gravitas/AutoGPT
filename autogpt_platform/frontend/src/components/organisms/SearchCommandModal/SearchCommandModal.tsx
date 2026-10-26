import { Text } from "@/components/atoms/Text/Text";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Separator } from "@/components/atoms/Separator/Separator";
import { cn } from "@/lib/utils";
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
import { Cancel01Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { isKey } from "@/lib/keyboard";

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
    if (isKey(event, "ArrowDown")) {
      event.preventDefault();
      moveHighlight(1);
    } else if (isKey(event, "ArrowUp")) {
      event.preventDefault();
      moveHighlight(-1);
    } else if (isKey(event, "Enter")) {
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
          className="fixed left-1/2 top-[18vh] z-[80] w-[calc(100%-2rem)] max-w-xl -translate-x-1/2 overflow-hidden rounded-3xl bg-surface shadow-2xl ring-1 ring-border-base focus:outline-none"
          onKeyDown={handleKeyDown}
        >
          <RXDialog.Title className="sr-only">Search</RXDialog.Title>
          <RXDialog.Description className="sr-only">
            Search commands and results.
          </RXDialog.Description>
          <div
            data-slot="search-command-modal-search-bar"
            className="flex items-center gap-3 bg-surface-muted p-3"
          >
            <Icon
              icon={Search01Icon}
              className="h-5 w-5 shrink-0 text-foreground-emphasis"
            />
            <Input
              bare
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
              className="h-9 border-0 bg-transparent px-0 text-foreground shadow-none placeholder:text-foreground-body focus-visible:ring-0"
            />
            {isLoading && isSearching ? (
              <LoadingSpinner
                size="small"
                aria-label="Searching"
                className="shrink-0 text-foreground-tertiary"
              />
            ) : null}
            {query ? (
              <Button
                type="button"
                variant="compact-ghost"
                size="compact-icon-sm"
                aria-label="Clear search"
                onClick={() => onQueryChange("")}
                className="shrink-0"
              >
                <Icon icon={Cancel01Icon} className="h-4 w-4" />
              </Button>
            ) : null}
          </div>
          <Separator />
          <div data-slot="search-command-modal-body" className="py-2">
            <div
              id={`${idPrefix}-results`}
              data-slot="search-command-modal-results"
              className={cn(
                "max-h-[26rem] overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border-base",
                totalCount === 0 && "max-h-none",
              )}
            >
              {isError ? (
                <Text
                  variant="body-tight"
                  as="div"
                  tone="error"
                  data-slot="search-command-modal-error"
                  className="px-3 py-8 text-center"
                >
                  {errorLabel}
                </Text>
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
                <Text
                  variant="body-tight"
                  as="div"
                  tone="muted"
                  data-slot="search-command-modal-empty"
                  className="px-3 py-8 text-center"
                >
                  {isSearching ? searchingEmptyLabel : idleEmptyLabel}
                </Text>
              ) : null}
            </div>
          </div>
          <Separator />
          <div
            data-slot="search-command-modal-footer"
            className="flex items-center justify-between gap-4 bg-surface-muted px-4 py-4"
          >
            <div
              data-slot="search-command-modal-hints"
              className="flex items-center gap-4"
            >
              <div
                data-slot="search-command-modal-hint"
                className="flex items-center gap-1.5"
              >
                <Text
                  variant="caption-sm"
                  as="kbd"
                  tone="emphasis"
                  data-slot="search-command-modal-key"
                  // TODO(tokens): no shadow token for the key-cap shadow.
                  className="inline-flex h-5 w-5 items-center justify-center rounded-md border border-border-base bg-surface shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)]"
                >
                  ↑
                </Text>
                <Text
                  variant="caption-sm"
                  as="kbd"
                  tone="emphasis"
                  data-slot="search-command-modal-key"
                  // TODO(tokens): no shadow token for the key-cap shadow.
                  className="inline-flex h-5 w-5 items-center justify-center rounded-md border border-border-base bg-surface shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)]"
                >
                  ↓
                </Text>
                <Text
                  variant="small-tight"
                  as="span"
                  tone="body"
                  data-slot="search-command-modal-hint-label"
                >
                  Navigate
                </Text>
              </div>
              <div
                data-slot="search-command-modal-hint"
                className="flex items-center gap-1.5"
              >
                <Text
                  variant="caption-sm"
                  as="kbd"
                  tone="emphasis"
                  data-slot="search-command-modal-key"
                  // TODO(tokens): no shadow token for the key-cap shadow.
                  className="inline-flex h-5 min-w-5 items-center justify-center rounded-md border border-border-base bg-surface px-1 shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)]"
                >
                  ↵
                </Text>
                <Text
                  variant="small-tight"
                  as="span"
                  tone="body"
                  data-slot="search-command-modal-hint-label"
                >
                  Select
                </Text>
              </div>
            </div>
            <div
              data-slot="search-command-modal-hint"
              className="flex items-center gap-1.5"
            >
              <Text
                variant="caption-sm"
                as="kbd"
                tone="emphasis"
                data-slot="search-command-modal-key"
                // TODO(tokens): no shadow token for the key-cap shadow.
                className="inline-flex h-5 items-center justify-center rounded-md border border-border-base bg-surface px-1.5 shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)]"
              >
                esc
              </Text>
              <Text
                variant="small-tight"
                as="span"
                tone="body"
                data-slot="search-command-modal-hint-label"
              >
                Close
              </Text>
            </div>
          </div>
        </RXDialog.Content>
      </RXDialog.Portal>
    </RXDialog.Root>
  );
}
