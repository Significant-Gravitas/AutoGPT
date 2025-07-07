"use client";

import { Button } from "@/components/ui/button";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";

function PaginationControlsContent({
  currentPage,
  totalPages,
  pathParam = "page",
}: {
  currentPage: number;
  totalPages: number;
  pathParam?: string;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const createPageUrl = (page: number) => {
    const params = new URLSearchParams(searchParams);
    params.set(pathParam, page.toString());
    return `${pathname}?${params.toString()}`;
  };

  const handlePageChange = (page: number) => {
    router.push(createPageUrl(page));
  };

  return (
    <div className="mt-4 flex items-center justify-center space-x-2">
      <Button
        variant="outline"
        onClick={() => handlePageChange(currentPage - 1)}
        disabled={currentPage <= 1}
      >
        Previous
      </Button>
      <span className="text-sm">
        Page {currentPage} of {totalPages}
      </span>
      <Button
        variant="outline"
        onClick={() => handlePageChange(currentPage + 1)}
        disabled={currentPage >= totalPages}
      >
        Next
      </Button>
    </div>
  );
}

export function PaginationControls({
  currentPage,
  totalPages,
  pathParam = "page",
}: {
  currentPage: number;
  totalPages: number;
  pathParam?: string;
}) {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center">Loading...</div>
      }
    >
      <PaginationControlsContent
        currentPage={currentPage}
        totalPages={totalPages}
        pathParam={pathParam}
      />
    </Suspense>
  );
}
