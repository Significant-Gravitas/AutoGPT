"use client";

import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { Button } from "@/components/__legacy__/ui/button";

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
      <Button
        variant="outline"
        data-slot="pagination-controls-previous"
        onClick={() => handlePageChange(currentPage - 1)}
        disabled={currentPage <= 1}
      >
        Previous
      </Button>
      <span
        data-slot="pagination-controls-label"
        className="text-body leading-5"
      >
        Page {currentPage} of {totalPages}
      </span>
      <Button
        variant="outline"
        data-slot="pagination-controls-next"
        onClick={() => handlePageChange(currentPage + 1)}
        disabled={currentPage >= totalPages}
      >
        Next
      </Button>
    </div>
  );
}
