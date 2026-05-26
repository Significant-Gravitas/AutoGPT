import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { MutableRefObject } from "react";
import { highlightMatch, type SearchCommandItem } from "./helpers";

const indicatorTransition = {
  type: "spring" as const,
  damping: 30,
  stiffness: 520,
  mass: 0.8,
};

interface Props {
  item: SearchCommandItem;
  /** Unique DOM id prefix so multiple modals on a page never collide. */
  idPrefix: string;
  query: string;
  isHighlighted: boolean;
  highlightedRef?: MutableRefObject<HTMLButtonElement | null>;
  onHighlight: () => void;
  onSelect: () => void;
}

export function SearchCommandResultItem({
  item,
  idPrefix,
  query,
  isHighlighted,
  highlightedRef,
  onHighlight,
  onSelect,
}: Props) {
  const reduceMotion = useReducedMotion();
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
        "relative h-auto w-full justify-start rounded-md px-3 py-2 text-left hover:bg-transparent",
      )}
    >
      {isHighlighted && (
        <>
          <motion.div
            layoutId={`${idPrefix}-highlight`}
            aria-hidden="true"
            className="absolute inset-0 z-0 rounded-md bg-zinc-100"
            transition={reduceMotion ? { duration: 0 } : indicatorTransition}
          />
          <motion.div
            layoutId={`${idPrefix}-highlight-bar`}
            aria-hidden="true"
            className="absolute inset-y-0 left-0 z-[1] my-auto h-5 w-[3px] rounded-full bg-zinc-900"
            transition={reduceMotion ? { duration: 0 } : indicatorTransition}
          />
        </>
      )}
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
        <AnimatePresence initial={false}>
          {isHighlighted && (
            <motion.span
              aria-hidden="true"
              initial={
                reduceMotion
                  ? { opacity: 0 }
                  : { opacity: 0, scale: 0.95, x: 80 }
              }
              animate={
                reduceMotion ? { opacity: 1 } : { opacity: 1, scale: 1, x: 0 }
              }
              exit={
                reduceMotion
                  ? { opacity: 0 }
                  : { opacity: 0, scale: 0.95, x: 80 }
              }
              transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
              style={{ willChange: "transform, opacity" }}
              className="inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-md border border-zinc-200 bg-white px-1 font-sans text-[11px] text-zinc-600 shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)]"
            >
              ↵
            </motion.span>
          )}
        </AnimatePresence>
      </div>
    </Button>
  );
}
