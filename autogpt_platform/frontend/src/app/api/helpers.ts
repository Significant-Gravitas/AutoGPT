import type { InfiniteData } from "@tanstack/react-query";
}

/** Make one list from a paginated infinite query result. */
export function unpaginate<
  TResponse extends { status: number; data: any },
  TPageDataKey extends {
    // Only allow keys for which the value is an array:
    [K in keyof OKData<TResponse>]: OKData<TResponse>[K] extends any[]
      ? K
      : never;
  }[keyof OKData<TResponse>] &
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
    typeof (page as { data: unknown }).data === "object" &&
    (page as { data: unknown }).data !== null &&
    "pagination" in (page as { data: Record<string, unknown> }).data &&
    typeof (page as { data: { pagination: unknown } }).data.pagination ===
      "object"
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
