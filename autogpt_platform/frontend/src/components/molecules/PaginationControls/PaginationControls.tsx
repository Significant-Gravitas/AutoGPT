"use client";

import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { Text } from "@/components/atoms/Text/Text";

// Rounded-md outline buttons; the Button atom's outline variant is a pill.
const pageButtonClassName =
  "inline-flex h-9 items-center justify-center whitespace-nowrap rounded-md border border-border-base bg-surface px-4 py-2 text-body font-medium leading-5 shadow-sm transition-colors hover:bg-surface-hover hover:text-foreground-strong focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50";

interface Props {
  currentPage: number;
  totalPages: number;
  pathParam?: string;
}

export function PaginationControls({
  currentPage,
  totalPages,
  pathParam = "page",
}: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  function createPageUrl(page: number) {
    const params = new URLSearchParams(searchParams);
    params.set(pathParam, page.toString());
    return `${pathname}?${params.toString()}`;
  }

  function handlePageChange(page: number) {
    router.push(createPageUrl(page));
  }

  return (
    <div
      data-slot="pagination-controls"
      className="mt-4 flex items-center justify-center space-x-2"
    >
      <button
        type="button"
        data-slot="pagination-controls-previous"
        className={pageButtonClassName}
        onClick={() => handlePageChange(currentPage - 1)}
        disabled={currentPage <= 1}
      >
        Previous
      </button>
      <Text
        variant="body-tight"
        as="span"
        tone="inherit"
        data-slot="pagination-controls-label"
      >
        Page {currentPage} of {totalPages}
      </Text>
      <button
        type="button"
        data-slot="pagination-controls-next"
        className={pageButtonClassName}
        onClick={() => handlePageChange(currentPage + 1)}
        disabled={currentPage >= totalPages}
      >
        Next
      </button>
    </div>
  );
}
