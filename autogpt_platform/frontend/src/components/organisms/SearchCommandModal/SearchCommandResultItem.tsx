import { Kbd } from "@/components/atoms/Kbd/Kbd";
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
}

export function SearchCommandResultItem({
  item,
  idPrefix,
  query,
  isHighlighted,
  isLoading = false,
  highlightedRef,
  onHighlight,
  onSelect,
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
      className={cn(
        "relative h-auto w-full justify-start rounded-md px-3 py-2 text-left transition-colors duration-150",
        isHighlighted ? "bg-muted hover:bg-muted" : "hover:bg-muted/50",
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
                isHighlighted ? "text-foreground" : "text-muted-foreground",
              )}
            />
          </span>
        ) : null}
        <div className="min-w-0 flex-1">
          <div
            className={cn(
              "truncate text-sm font-normal transition-colors duration-150",
              isHighlighted ? "text-foreground" : "text-foreground",
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
            <div className="mt-0.5 truncate text-xs text-muted-foreground">
              {item.subtitle}
            </div>
          ) : null}
        </div>
        {isLoading ? (
          <LoadingSpinner
            size="small"
            aria-label="Opening"
            className="shrink-0 text-muted-foreground"
          />
        ) : (
          <Kbd
            aria-hidden="true"
            className={cn(
              "text-muted-foreground transition-opacity duration-150",
              isHighlighted ? "opacity-100" : "opacity-0",
            )}
          >
            ↵
          </Kbd>
        )}
      </div>
    </Button>
  );
}
