"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Pagination as KobraPagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
} from "@/components/ui/pagination";
import { cn } from "@/lib/utils";
import { ArrowLeft01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import type { Ref } from "react";
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
  ref?: Ref<HTMLElement>;
}

export function Pagination({
  page,
  pageCount,
  onPageChange,
  siblingCount = 1,
  disabled = false,
  className,
  "aria-label": ariaLabel = "Pagination",
  ref,
}: Props) {
  const items = getPageItems(page, pageCount, siblingCount);

  function goTo(next: number) {
    if (next < 1 || next > pageCount || next === page) return;
    onPageChange(next);
  }

  return (
    <KobraPagination ref={ref} aria-label={ariaLabel} className={className}>
      <PaginationContent className="gap-1">
        <PaginationItem>
          <Button
            type="button"
            variant="ghost"
            size="md"
            leadingIcon={ArrowLeft01Icon}
            onClick={() => goTo(page - 1)}
            disabled={disabled || page <= 1}
            className="min-w-0 px-2.5"
          >
            Previous
          </Button>
        </PaginationItem>
        {items.map((item) =>
          typeof item === "number" ? (
            <PaginationItem key={item}>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                withTooltip={false}
                onClick={() => goTo(item)}
                disabled={disabled}
                aria-label={`Page ${item}`}
                aria-current={item === page ? "page" : undefined}
                data-active={item === page}
                className={cn(
                  "rounded-lg text-sm font-medium text-muted-foreground",
                  item === page && "text-foreground",
                )}
              >
                {item}
              </Button>
            </PaginationItem>
          ) : (
            <PaginationItem key={item}>
              <PaginationEllipsis className="text-muted-foreground" />
            </PaginationItem>
          ),
        )}
        <PaginationItem>
          <Button
            type="button"
            variant="ghost"
            size="md"
            rightIcon={<Icon icon={ArrowRight01Icon} size={16} aria-hidden />}
            onClick={() => goTo(page + 1)}
            disabled={disabled || page >= pageCount}
            className="min-w-0 px-2.5"
          >
            Next
          </Button>
        </PaginationItem>
      </PaginationContent>
    </KobraPagination>
  );
}
