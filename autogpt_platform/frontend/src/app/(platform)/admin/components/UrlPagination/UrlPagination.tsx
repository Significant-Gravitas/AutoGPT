"use client";

import { Pagination } from "@/components/molecules/Pagination/Pagination";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

interface Props {
  page: number;
  pageCount: number;
  /** Search parameter that holds the page number. */
  param?: string;
}

// Pagination for server-rendered admin tables: the page lives in the URL.
export function UrlPagination({ page, pageCount, param = "page" }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  function handlePageChange(next: number) {
    const params = new URLSearchParams(searchParams);
    params.set(param, String(next));
    router.push(`${pathname}?${params.toString()}`);
  }

  return (
    <Pagination
      page={page}
      pageCount={pageCount}
      onPageChange={handlePageChange}
      className="mt-4"
    />
  );
}
