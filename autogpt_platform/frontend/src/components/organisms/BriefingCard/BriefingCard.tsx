"use client";

import { motion, useReducedMotion } from "framer-motion";
import {
  ArrowDown01Icon,
  ArrowDown02Icon,
  ArrowUp02Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { BriefingResponse } from "@/app/api/__generated__/models/briefingResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { RunRow } from "./components/RunRow";
import { formatBriefingDate } from "./helpers";
import { COLLAPSED_ROWS, useBriefingCard } from "./useBriefingCard";

interface Props {
  briefing: BriefingResponse;
  className?: string;
}

// The briefing's decision items are deliberately not rendered here: the same
// pending reviews already appear — and are actionable — in the
// needs-attention list on the home page. They stay in the thread markdown,
// where there is no such list.
export function BriefingCard({ briefing, className }: Props) {
  const {
    listRef,
    height,
    isShowingAll,
    toggleShowAll,
    canScrollUp,
    canScrollDown,
    scrollByStep,
  } = useBriefingCard();
  const shouldReduceMotion = useReducedMotion();
  const { run_items } = briefing.content;

  // A briefing can be all decisions and no terminal runs (a run paused on an
  // approval never completes), which would render as a card containing just
  // a date. The needs-attention list carries that case on its own.
  if (run_items.length === 0) return null;

  const hasMore = run_items.length > COLLAPSED_ROWS;
  const transition = shouldReduceMotion
    ? { duration: 0 }
    : { duration: 0.32, ease: [0.32, 0.72, 0, 1] as const };

  return (
    <section className={cn("text-left", className)}>
      <div className="mb-2 flex items-center gap-2 px-2">
        <Icon icon={SparklesIcon} size={16} className="text-zinc-400" />
        <Text variant="body" className="text-zinc-700">
          Recap
        </Text>
        <Text variant="body" className="text-zinc-400">
          {formatBriefingDate(briefing.briefing_date)}
        </Text>
      </div>

      <div className="overflow-hidden rounded-3xl bg-white shadow-zinc-950 smooth-shadow-ring-sm">
        <motion.div
          // Real height, not a layout transform: the page below has to reflow
          // with the card, and a transform would scale the rows' text.
          initial={false}
          animate={{ height: height ?? "auto" }}
          transition={transition}
          className="relative"
        >
          <ul
            ref={listRef}
            className={cn(
              "h-full divide-y divide-zinc-100",
              isShowingAll
                ? "overflow-y-auto scrollbar-none"
                : "overflow-hidden",
            )}
          >
            {run_items.map((item) => (
              <RunRow key={item.execution_id} item={item} />
            ))}
          </ul>

          {/* Inside the card, over the scrolling rows: the scrollbar is
              hidden, so these are the only affordance once the list runs past
              its window. */}
          <ScrollArrow
            direction="up"
            isVisible={canScrollUp}
            onScroll={() => scrollByStep(-1)}
          />
          <ScrollArrow
            direction="down"
            isVisible={canScrollDown}
            onScroll={() => scrollByStep(1)}
          />
        </motion.div>

        {hasMore ? (
          <button
            type="button"
            onClick={toggleShowAll}
            className="flex w-full items-center justify-center gap-1.5 border-t border-zinc-100 px-4 py-3 text-zinc-500 transition-colors hover:bg-zinc-50 hover:text-zinc-900"
          >
            <Text variant="small-medium" className="text-inherit">
              {isShowingAll
                ? "Show less"
                : `Show all results (${run_items.length})`}
            </Text>
            <Icon
              icon={ArrowDown01Icon}
              size={14}
              className={cn(
                "transition-transform",
                isShowingAll && "rotate-180",
              )}
            />
          </button>
        ) : null}
      </div>
    </section>
  );
}

function ScrollArrow({
  direction,
  isVisible,
  onScroll,
}: {
  direction: "up" | "down";
  isVisible: boolean;
  onScroll: () => void;
}) {
  const isUp = direction === "up";

  return (
    <motion.div
      initial={false}
      animate={{ opacity: isVisible ? 1 : 0 }}
      transition={{ duration: 0.15 }}
      className={cn(
        "pointer-events-none absolute inset-x-0 flex h-12 items-center justify-center",
        isUp
          ? "top-0 bg-gradient-to-b from-white via-white/85 to-transparent"
          : "bottom-0 bg-gradient-to-t from-white via-white/85 to-transparent",
      )}
    >
      <button
        type="button"
        onClick={onScroll}
        tabIndex={isVisible ? 0 : -1}
        aria-hidden={!isVisible}
        aria-label={isUp ? "Scroll up" : "Scroll down"}
        className={cn(
          "flex size-8 items-center justify-center rounded-full bg-white text-zinc-500 shadow-zinc-950 transition-colors smooth-shadow-ring-sm hover:text-zinc-900",
          isVisible && "pointer-events-auto",
        )}
      >
        <Icon icon={isUp ? ArrowUp02Icon : ArrowDown02Icon} size={16} />
      </button>
    </motion.div>
  );
}
