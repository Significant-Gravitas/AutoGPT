"use client";

import { CaretLeftIcon, CaretRightIcon } from "@phosphor-icons/react";

import type { Pagination as PaginationModel } from "@/app/api/__generated__/models/pagination";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  pagination: PaginationModel;
  onPageChange: (page: number) => void;
  disabled?: boolean;
}

const MAX_VISIBLE_PAGES = 5;

function getVisiblePages(current: number, total: number): number[] {
  if (total <= MAX_VISIBLE_PAGES) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }
  let start = Math.max(1, current - Math.floor(MAX_VISIBLE_PAGES / 2));
  let end = start + MAX_VISIBLE_PAGES - 1;
  if (end > total) {
    end = total;
    start = end - MAX_VISIBLE_PAGES + 1;
  }
  return Array.from({ length: end - start + 1 }, (_, i) => start + i);
}

export function Pagination({ pagination, onPageChange, disabled }: Props) {
  const { current_page, total_pages, total_items, page_size } = pagination;

  if (total_pages <= 1) return null;

  const startItem = (current_page - 1) * page_size + 1;
  const endItem = Math.min(current_page * page_size, total_items);
  const visiblePages = getVisiblePages(current_page, total_pages);

  return (
    <div
      className="flex items-center justify-between gap-3 px-4 py-3"
      data-testid="submissions-pagination"
    >
      <Text variant="small" className="text-zinc-500">
        Showing {startItem}–{endItem} of {total_items}
      </Text>
      <div className="flex items-center gap-1.5">
        <Button
          variant="secondary"
          size="small"
          leftIcon={<CaretLeftIcon size={14} />}
          disabled={disabled || current_page <= 1}
          onClick={() => onPageChange(current_page - 1)}
          aria-label="Previous page"
        >
          Previous
        </Button>
        <div className="flex items-center gap-1">
          {visiblePages.map((page) => {
            const isActive = page === current_page;
            return (
              <button
                key={page}
                type="button"
                disabled={disabled}
                onClick={() => onPageChange(page)}
                aria-label={`Go to page ${page}`}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "ease-[cubic-bezier(0.16,1,0.3,1)] inline-flex h-8 min-w-8 items-center justify-center rounded-full px-2 text-xs font-medium tabular-nums transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50",
                  isActive
                    ? "bg-zinc-900 text-white"
                    : "text-zinc-700 hover:bg-zinc-100",
                )}
              >
                {page}
              </button>
            );
          })}
        </div>
        <Button
          variant="secondary"
          size="small"
          rightIcon={<CaretRightIcon size={14} />}
          disabled={disabled || current_page >= total_pages}
          onClick={() => onPageChange(current_page + 1)}
          aria-label="Next page"
        >
          Next
        </Button>
      </div>
    </div>
  );
}
