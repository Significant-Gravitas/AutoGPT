"use client";

import { motion, useReducedMotion } from "framer-motion";
import {
  ArrowDown01Icon,
  ArrowDown02Icon,
  ArrowUp02Icon,
  News01Icon,
} from "@hugeicons/core-free-icons";
import type { BriefingResponse } from "@/app/api/__generated__/models/briefingResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AttentionCard } from "@/components/organisms/AttentionCard/AttentionCard";
import { cn } from "@/lib/utils";
import { RunRow } from "./components/RunRow";
import { COLLAPSED_ROWS, formatBriefingDate, hasRecapContent } from "./helpers";
import { useBriefingCard } from "./useBriefingCard";

interface Props {
  briefing: BriefingResponse;
  className?: string;
}

// The briefing's decision items are deliberately not rendered here: the same
// pending reviews already appear — and are actionable — in the
// needs-attention list on the home page. They stay in the thread markdown,
// where there is no such list.
export function BriefingCard({ briefing, className }: Props) {
  const { run_items } = briefing.content;
  const {
    listRef,
    height,
    heightTransition,
    isShowingAll,
    toggleShowAll,
    canScrollUp,
    canScrollDown,
    scrollByStep,
  } = useBriefingCard(run_items.map((item) => item.execution_id).join("|"));

  // Callers gate on hasRecapContent so they can show their own fallback;
  // this keeps the organism safe to mount on its own.
  if (!hasRecapContent(briefing)) return null;

  const hasMore = run_items.length > COLLAPSED_ROWS;

  return (
    <section data-slot="briefing-card" className={cn("text-left", className)}>
      <div
        data-slot="briefing-card-header"
        className="mb-2 flex items-center gap-2 px-2"
      >
        <Icon icon={News01Icon} size={16} className="text-foreground-subtle" />
        <Text variant="body" className="text-foreground-body">
          Recap
        </Text>
        {/* Both stops stay at or below purple-600 so the lightest painted
            pixel still clears WCAG AA (4.5:1) against the white card. */}
        <Text
          variant="body"
          // TODO(tokens): no role for the purple-800 to purple-600 gradient
          // stops.
          className="bg-gradient-to-r from-purple-800 to-purple-600 bg-clip-text text-transparent"
        >
          {formatBriefingDate(briefing.briefing_date)}
        </Text>
      </div>

      <AttentionCard.Root
        variant="elevated"
        density="flush"
        data-slot="briefing-card-panel"
      >
        <motion.div
          // Real height, not a layout transform: the page below has to reflow
          // with the card, and a transform would scale the rows' text.
          initial={false}
          animate={{ height: height ?? "auto" }}
          transition={heightTransition}
          data-slot="briefing-card-viewport"
          className="relative"
        >
          <ul
            ref={listRef}
            data-slot="briefing-card-list"
            className={cn(
              "h-full divide-y divide-separator",
              isShowingAll
                ? "overflow-y-auto scrollbar-none"
                : "overflow-hidden",
            )}
          >
            {run_items.map((item, index) => (
              <RunRow
                key={item.execution_id}
                item={item}
                isHidden={!isShowingAll && index >= COLLAPSED_ROWS}
              />
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
            data-slot="briefing-card-toggle"
            // The hairline stays the button's own top border: a separate
            // Separator would move the button, and its focus outline, 1px.
            className="flex w-full items-center justify-center gap-1.5 border-t border-separator px-4 py-3 text-foreground-tertiary transition-colors hover:bg-surface-muted hover:text-foreground-strong"
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
      </AttentionCard.Root>
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
  const shouldReduceMotion = useReducedMotion();

  return (
    <motion.div
      initial={false}
      animate={{ opacity: isVisible ? 1 : 0 }}
      transition={{ duration: shouldReduceMotion ? 0 : 0.15 }}
      data-slot="briefing-card-scroll-arrow"
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
        data-slot="briefing-card-scroll-arrow-control"
        className={cn(
          // TODO(tokens): no shadow-colour role for zinc-950 (#09090B).
          "flex size-8 items-center justify-center rounded-full bg-surface text-foreground-tertiary shadow-zinc-950 transition-colors smooth-shadow-ring-sm hover:text-foreground-strong",
          isVisible && "pointer-events-auto",
        )}
      >
        <Icon icon={isUp ? ArrowUp02Icon : ArrowDown02Icon} size={16} />
      </button>
    </motion.div>
  );
}
