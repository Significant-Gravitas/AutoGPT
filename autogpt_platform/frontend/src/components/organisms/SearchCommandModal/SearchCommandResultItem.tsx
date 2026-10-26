import { Text } from "@/components/atoms/Text/Text";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Button } from "@/components/atoms/Button/Button";
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
      variant="compact-ghost"
      size="compact"
      role="option"
      aria-selected={isHighlighted}
      onMouseEnter={onHighlight}
      onClick={onSelect}
      className={cn(
        "relative h-auto w-full justify-start rounded-md px-3 py-2 text-left transition-colors duration-150",
        isHighlighted
          ? "bg-surface-hover hover:bg-surface-hover"
          : "hover:bg-surface-muted",
      )}
    >
      <span
        aria-hidden="true"
        data-slot="search-command-result-item-indicator"
        className={cn(
          "absolute inset-y-0 left-0 my-auto h-5 w-[3px] rounded-full bg-action-hover transition-opacity duration-150",
          isHighlighted ? "opacity-100" : "opacity-0",
        )}
      />
      <div
        data-slot="search-command-result-item-body"
        className="relative z-10 flex min-w-0 flex-1 items-center gap-2.5"
      >
        {Icon ? (
          <span
            aria-hidden
            data-slot="search-command-result-item-icon"
            className="contents"
          >
            <Icon
              className={cn(
                "h-4 w-4 shrink-0 transition-colors duration-150",
                isHighlighted
                  ? "text-foreground-strong"
                  : "text-foreground-tertiary",
              )}
            />
          </span>
        ) : null}
        <div
          data-slot="search-command-result-item-content"
          className="min-w-0 flex-1"
        >
          <Text
            variant="body-tight"
            as="div"
            data-slot="search-command-result-item-title"
            className={cn(
              "truncate transition-colors duration-150",
              isHighlighted ? "text-foreground" : "text-foreground-emphasis",
            )}
          >
            {highlightMatch(item.title, query).map((part, partIndex) => (
              <span
                key={`${part.text}-${partIndex}`}
                data-slot="search-command-result-item-title-segment"
                className={part.isMatch ? "font-semibold" : undefined}
              >
                {part.text}
              </span>
            ))}
          </Text>
          {item.subtitle ? (
            <Text
              variant="small-medium-tight"
              as="div"
              tone="muted"
              data-slot="search-command-result-item-subtitle"
              className="mt-0.5 truncate"
            >
              {item.subtitle}
            </Text>
          ) : null}
        </div>
        {isLoading ? (
          <LoadingSpinner
            size="small"
            aria-label="Opening"
            className="shrink-0 text-foreground-tertiary"
          />
        ) : (
          <span
            aria-hidden="true"
            data-slot="search-command-result-item-key"
            className={cn(
              // TODO(tokens): no shadow token for the key-cap shadow.
              "inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-md border border-border-base bg-surface px-1 font-sans text-caption-sm leading-inherit text-foreground-secondary shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)] transition-opacity duration-150",
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
