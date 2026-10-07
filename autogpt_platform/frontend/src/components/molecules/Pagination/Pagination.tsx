"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { forwardRef } from "react";
import { getPageItems } from "./helpers";

interface Props {
  /** Current page, starting at 1. */
  page: number;
  pageCount: number;
  onPageChange: (page: number) => void;
  /** Pages shown either side of the current one before an ellipsis. */
  siblingCount?: number;
  disabled?: boolean;
  className?: string;
  "aria-label"?: string;
}

export const Pagination = forwardRef<HTMLElement, Props>(function Pagination(
  {
    page,
    pageCount,
    onPageChange,
    siblingCount = 1,
    disabled = false,
    className,
    "aria-label": ariaLabel = "Pagination",
  },
  ref,
) {
  const items = getPageItems(page, pageCount, siblingCount);
  const focusRing =
    "focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:ring-offset-2";

  function goTo(next: number) {
    if (next < 1 || next > pageCount || next === page) return;
    onPageChange(next);
  }

  return (
    <nav
      ref={ref}
      aria-label={ariaLabel}
      className={cn("flex justify-center", className)}
    >
      <ul className="flex items-center gap-1">
        <li>
          <Button
            type="button"
            variant="ghost"
            size="md"
            leadingIcon={ArrowLeft01Icon}
            onClick={() => goTo(page - 1)}
            disabled={disabled || page <= 1}
            className={cn("min-w-0 px-2.5", focusRing)}
          >
            Previous
          </Button>
        </li>
        {items.map((item) =>
          typeof item === "number" ? (
            <li key={item}>
              <Button
                type="button"
                variant={item === page ? "primary" : "ghost"}
                size="icon-sm"
                withTooltip={false}
                onClick={() => goTo(item)}
                disabled={disabled}
                aria-label={`Page ${item}`}
                aria-current={item === page ? "page" : undefined}
                className={cn("text-sm font-medium", focusRing)}
              >
                {item}
              </Button>
            </li>
          ) : (
            <li
              key={item}
              aria-hidden
              className="flex size-8 items-center justify-center text-zinc-500"
            >
              <Icon icon={MoreHorizontalIcon} size={16} />
            </li>
          ),
        )}
        <li>
          <Button
            type="button"
            variant="ghost"
            size="md"
            rightIcon={<Icon icon={ArrowRight01Icon} size={16} aria-hidden />}
            onClick={() => goTo(page + 1)}
            disabled={disabled || page >= pageCount}
            className={cn("min-w-0 px-2.5", focusRing)}
          >
            Next
          </Button>
        </li>
      </ul>
    </nav>
  );
});
