"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { Cancel01Icon, Search01Icon } from "@hugeicons/core-free-icons";
import {
  Command as CommandRoot,
  CommandInput,
  type CommandRef,
  type FilterFunctionType,
} from "@kmenu/react";
import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import {
  getTotalCount,
  toCommandOptions,
  type SearchCommandBucket,
  type SearchCommandItem,
  type SearchCommandOption,
  type SearchCommandOptionData,
} from "./helpers";
import { SearchCommandResults } from "./SearchCommandResults";
import { SearchCommandSkeleton } from "./SearchCommandSkeleton";

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

// The results are already filtered by whoever owns ``buckets`` (usually a
// server search), so kmenu must not filter them again.
const showEverything: FilterFunctionType<SearchCommandOptionData> = (options) =>
  options;

const headerButtonClassName =
  "flex cursor-pointer items-center rounded-md border border-border px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground";

const emptyClassName =
  "flex h-24 items-center justify-center text-sm text-muted-foreground";

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
  const titleId = useId();
  const descriptionId = useId();
  const commandRef = useRef<CommandRef<SearchCommandOptionData>>(null);
  // kmenu wires its callbacks once on mount, so they read the latest props
  // through refs instead of the closures it captured.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const onSelectItemRef = useRef(onSelectItem);
  onSelectItemRef.current = onSelectItem;
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!isOpen) {
      setIsReady(false);
      return;
    }
    const timeout = window.setTimeout(() => setIsReady(true), 180);
    return () => window.clearTimeout(timeout);
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    // Escape pressed with focus outside the dialog still closes it.
    function handleWindowKeyDown(event: globalThis.KeyboardEvent) {
      if (event.defaultPrevented || !isKey(event, "Escape")) return;
      event.preventDefault();
      onCloseRef.current();
    }
    window.addEventListener("keydown", handleWindowKeyDown);
    return () => window.removeEventListener("keydown", handleWindowKeyDown);
  }, [isOpen]);

  if (!isOpen) return null;

  const options = toCommandOptions(buckets);
  const totalCount = getTotalCount(buckets);
  const isSearching = query.trim().length > 0;

  function handleSelect(option: SearchCommandOption) {
    if (!option.data) return;
    onSelectItemRef.current(option.data.item, option.data.bucketKey);
  }

  // Closing is this component's job (kmenu would only park its core in an
  // idle state), so Escape is taken in the capture phase before kmenu sees it.
  function handleEscapeCapture(event: KeyboardEvent<HTMLDivElement>) {
    if (!isKey(event, "Escape")) return;
    event.preventDefault();
    event.stopPropagation();
    onClose();
  }

  function handleDialogKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    // Keys typed into the input are kmenu's; this covers focus elsewhere in
    // the dialog (the header buttons) so the list still navigates.
    if (event.target instanceof HTMLInputElement) return;
    const command = commandRef.current?.command;
    if (isKey(event, "ArrowDown")) {
      event.preventDefault();
      command?.navigateDown();
    } else if (isKey(event, "ArrowUp")) {
      event.preventDefault();
      command?.navigateUp();
    } else if (isKey(event, "Enter")) {
      event.preventDefault();
      command?.selectActive();
    }
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      aria-describedby={descriptionId}
      className="fixed inset-0 z-80 flex animate-in items-start justify-center bg-linear-to-b from-black/20 to-black/25 pt-[18vh] backdrop-blur-[2px] fade-in-0 motion-reduce:animate-none"
      onKeyDownCapture={handleEscapeCapture}
      onKeyDown={handleDialogKeyDown}
    >
      <button
        type="button"
        aria-label="Close search"
        className="fixed inset-0 cursor-default border-0 bg-transparent"
        onClick={onClose}
      />
      <div
        className={cn(
          "relative w-[90%] max-w-[620px] animate-in overflow-hidden rounded-[14px] border border-border bg-popover shadow-2xl duration-250 ease-[cubic-bezier(0.16,1,0.3,1)] fade-in-0 zoom-in-95 slide-in-from-top-1 motion-reduce:animate-none",
          // kmenu's active-row indicator (styled by Kobra's command-menu.css)
          // only starts gliding once the dialog has settled.
          isReady &&
            "[&_.command-active-indicator]:transition-[transform,width,height] [&_.command-active-indicator]:duration-150 [&_.command-active-indicator]:ease-[cubic-bezier(0.16,1,0.3,1)] motion-reduce:[&_.command-active-indicator]:transition-none",
        )}
      >
        <span id={titleId} className="sr-only">
          Search
        </span>
        <span id={descriptionId} className="sr-only">
          Search commands and results.
        </span>
        <CommandRoot
          ref={commandRef}
          open
          value={query}
          options={options}
          filter={showEverything}
          onSelect={handleSelect}
          className="flex flex-col"
        >
          <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
            <Icon
              icon={Search01Icon}
              className="size-5 shrink-0 text-muted-foreground"
            />
            {/* kmenu marks the input as a combobox; it stays a plain textbox
                with the listbox wired through aria-controls, as before. */}
            <CommandInput
              role={undefined}
              aria-expanded={undefined}
              value={query}
              aria-label={inputAriaLabel}
              placeholder={placeholder}
              autoComplete="off"
              className="min-w-0 flex-1 border-0 bg-transparent text-[0.95rem] text-popover-foreground outline-0 placeholder:text-muted-foreground"
              onValueChange={onQueryChange}
            />
            {isLoading && isSearching ? (
              <LoadingSpinner
                size="small"
                aria-label="Searching"
                className="shrink-0 text-muted-foreground"
              />
            ) : null}
            {query ? (
              <button
                type="button"
                aria-label="Clear search"
                className={headerButtonClassName}
                onClick={() => onQueryChange("")}
              >
                <Icon icon={Cancel01Icon} className="size-3" />
              </button>
            ) : null}
            <button
              type="button"
              className={headerButtonClassName}
              onClick={onClose}
            >
              Esc
            </button>
          </div>
          {isError ? (
            <div className={cn(emptyClassName, "text-destructive")}>
              {errorLabel}
            </div>
          ) : totalCount > 0 ? (
            <SearchCommandResults
              buckets={buckets}
              options={options}
              query={query.trim()}
              loadingItemId={loadingItemId}
            />
          ) : isLoading ? (
            // Skeleton over text: the input-side spinner already signals
            // "searching", so the body mirrors the shape of what's about to
            // appear rather than blocking the eye with a centred string.
            <div className="p-2">
              <SearchCommandSkeleton />
            </div>
          ) : (
            <div className={emptyClassName}>
              {isSearching ? searchingEmptyLabel : idleEmptyLabel}
            </div>
          )}
        </CommandRoot>
      </div>
    </div>
  );
}
