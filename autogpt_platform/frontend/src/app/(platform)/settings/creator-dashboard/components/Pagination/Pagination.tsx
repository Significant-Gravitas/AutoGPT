"use client";

import { CaretLeftIcon, CaretRightIcon } from "@phosphor-icons/react";

import type { Pagination as PaginationModel } from "@/app/api/__generated__/models/pagination";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  pagination: PaginationModel;
  onPageChange: (page: number) => void;
  disabled?: boolean;
}

export function Pagination({ pagination, onPageChange, disabled }: Props) {
  const { current_page, total_pages, total_items, page_size } = pagination;

  if (total_pages <= 1) return null;

  const startItem = (current_page - 1) * page_size + 1;
  const endItem = Math.min(current_page * page_size, total_items);

  return (
    <div
      className="flex items-center justify-between gap-3 px-4 py-3"
      data-testid="submissions-pagination"
    >
      <Text variant="small" className="text-zinc-500">
        Showing {startItem}–{endItem} of {total_items}
      </Text>
      <div className="flex items-center gap-2">
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
