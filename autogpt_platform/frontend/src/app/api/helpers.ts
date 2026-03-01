import type { InfiniteData } from "@tanstack/react-query";
    string,
  TItemData extends OKData<TResponse>[TPageDataKey][number],
>(
  infiniteData: InfiniteData<TResponse>,
  pageListKey: TPageDataKey,
): TItemData[] {
  return (
    infiniteData?.pages.flatMap<TItemData>((page) => {
      if (!hasValidListPage(page, pageListKey)) return [];
      return page.data[pageListKey] || [];
    }) || []
  );
}


function hasValidListPage<TKey extends string>(
  page: unknown,
  pageListKey: TKey,
): page is { status: 200; data: { [key in TKey]: any[] } } {
  return (
    typeof page === "object" &&
    page !== null &&
    "status" in page &&
    page.status === 200 &&
    "data" in page &&
    typeof page.data === "object" &&
    page.data !== null &&
    pageListKey in page.data &&
    Array.isArray((page.data as Record<string, unknown>)[pageListKey])
  );
}


export function hasValidPaginationInfo(
  page: unknown,
): page is { status: 200; data: { pagination: Pagination; [key: string]: any } } {
  return (
    typeof page === "object" &&
    page !== null &&
    "status" in page &&
    (page as { status: unknown }).status === 200 &&
    "data" in page &&
    typeof (page as { data: unknown }).data === "object" &&page.data !== null &&
    "pagination" in page.data &&
    typeof page.data.pagination === "object" &&
    page.data.pagination !== null &&
    "total_items" in page.data.pagination &&
    "total_pages" in page.data.pagination &&
    "current_page" in page.data.pagination &&
    "page_size" in page.data.pagination
  );
}


type ResponseWithData = { status: number; data: unknown };
type ExtractResponseData<T extends ResponseWithData> = T extends {
  data: infer D;
}
  ? D
  : never;
type SuccessfulResponses<T extends ResponseWithData> = T extends {
  status: infer S;
}
  ? S extends number
    ? `${S}` extends `2${string}`
      ? T
      : never
    : never
  : never;


/**
 * Resolve an Orval response to its payload after asserting the status is either the explicit
 * `expected` code or any other 2xx status if `expected` is omitted.
 *
 * Usage with server actions:
 * ```ts
