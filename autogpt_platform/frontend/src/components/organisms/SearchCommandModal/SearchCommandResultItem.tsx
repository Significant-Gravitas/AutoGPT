import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { MutableRefObject } from "react";
import { highlightMatch, type SearchCommandItem } from "./helpers";

interface Props {
  item: SearchCommandItem;
  /** Unique DOM id prefix so multiple modals on a page never collide. */
  idPrefix: string;
  query: string;
  isHighlighted: boolean;
  /** Shows a trailing spinner while the row's action is in-flight. */
  isLoading?: boolean;
  highlightedRef?: MutableRefObject<HTMLButtonElement | null>;
  onHighlight: () => void;
  onSelect: () => void;
  /** Zero-based position used to stagger the enter animation. */
  enterIndex?: number;
}

// Per Emil Kowalski's "faster is better" rule: keep the stagger tight
// so the list never feels like it's *waiting* to render. 35ms × 12
// items ≈ 420ms total before the last item finishes its own 180ms
// animation, which stays under the 300ms perceived-snappy budget for
// the typical first-visible row.
const ENTER_STAGGER_MS = 35;

export function SearchCommandResultItem({
  item,
  idPrefix,
  query,
  isHighlighted,
  isLoading = false,
  highlightedRef,
  onHighlight,
  onSelect,
  enterIndex = 0,
}: Props) {
  const Icon = item.icon;

  return (
    <Button
      ref={isHighlighted ? highlightedRef : undefined}
      id={`${idPrefix}-${item.id}`}
      type="button"
      variant="ghost"
      role="option"
      aria-selected={isHighlighted}
      onMouseEnter={onHighlight}
      onClick={onSelect}
      style={{ animationDelay: `${enterIndex * ENTER_STAGGER_MS}ms` }}
      className={cn(
        "relative h-auto w-full justify-start rounded-md px-3 py-2 text-left transition-colors duration-150",
        "motion-safe:animate-search-item-in",
        isHighlighted ? "bg-zinc-100 hover:bg-zinc-100" : "hover:bg-zinc-50",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "absolute inset-y-0 left-0 my-auto h-5 w-[3px] rounded-full bg-zinc-900 transition-opacity duration-150",
          isHighlighted ? "opacity-100" : "opacity-0",
        )}
      />
      <div className="relative z-10 flex min-w-0 flex-1 items-center gap-2.5">
        {Icon ? (
          <span aria-hidden className="contents">
            <Icon
              className={cn(
                "h-4 w-4 shrink-0 transition-colors duration-150",
                isHighlighted ? "text-zinc-900" : "text-zinc-500",
              )}
            />
          </span>
        ) : null}
        <div className="min-w-0 flex-1">
          <div
            className={cn(
              "truncate text-sm font-normal transition-colors duration-150",
              isHighlighted ? "text-zinc-950" : "text-zinc-800",
            )}
          >
            {highlightMatch(item.title, query).map((part, partIndex) => (
              <span
                key={`${part.text}-${partIndex}`}
                className={part.isMatch ? "font-semibold" : undefined}
              >
                {part.text}
              </span>
            ))}
          </div>
          {item.subtitle ? (
            <div className="mt-0.5 truncate text-xs text-zinc-500">
              {item.subtitle}
            </div>
          ) : null}
        </div>
        {isLoading ? (
          <LoadingSpinner
            size="small"
            aria-label="Opening"
            className="shrink-0 text-zinc-500"
          />
        ) : (
          <span
            aria-hidden="true"
            className={cn(
              "inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-md border border-zinc-200 bg-white px-1 font-sans text-[11px] text-zinc-600 shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)] transition-opacity duration-150",
              isHighlighted ? "opacity-100" : "opacity-0",
            )}
          >
            ↵
          </span>
        )}
      </div>
    </Button>
  );
}
