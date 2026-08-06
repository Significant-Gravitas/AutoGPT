"use client";

import { cn } from "@/lib/utils";
import { useRef, useState } from "react";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  text: string;
}

export function ExpertAboutSection({ text }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const { isOverflowing, measureRef } = useClampOverflow();
  // Once expanded the paragraph no longer overflows, so keep the toggle
  // visible from either signal.
  const showToggle = isOverflowing || isExpanded;

  return (
    <section>
      <div className="mb-2.5 text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
        About
      </div>
      <p
        ref={measureRef}
        className={cn(
          "max-w-prose whitespace-pre-line text-base leading-relaxed text-zinc-600",
          !isExpanded && "line-clamp-4",
        )}
      >
        {text}
      </p>
      {showToggle ? (
        <button
          type="button"
          onClick={() => setIsExpanded((v) => !v)}
          className="mt-2 flex items-center gap-1 text-sm font-medium text-zinc-500 transition-colors hover:text-zinc-900"
        >
          {isExpanded ? "Show less" : "Read more"}
          <Icon
            icon={ArrowDown01Icon}
            size={14}
            className={cn(
              "transition-transform duration-200",
              isExpanded && "rotate-180",
            )}
          />
        </button>
      ) : null}
    </section>
  );
}

/** Measures whether the clamped paragraph actually overflows, instead of
 *  guessing from character count — a short bio can still clamp on narrow or
 *  zoomed layouts. Re-measures on element resize. */
function useClampOverflow() {
  const [isOverflowing, setIsOverflowing] = useState(false);
  const observerRef = useRef<ResizeObserver | null>(null);

  function measureRef(node: HTMLParagraphElement | null) {
    observerRef.current?.disconnect();
    observerRef.current = null;
    if (!node) return;
    function measure() {
      if (!node) return;
      setIsOverflowing(node.scrollHeight > node.clientHeight + 1);
    }
    if (typeof ResizeObserver !== "undefined") {
      observerRef.current = new ResizeObserver(measure);
      observerRef.current.observe(node);
    }
    measure();
  }

  return { isOverflowing, measureRef };
}
