import type { InfiniteData } from "@tanstack/react-query";
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
      "object" &&
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
 *   const onboarding = await expectStatus(getV1OnboardingState());
 *   const agent = await expectStatus(
 *     postV2AddMarketplaceAgent({ store_listing_version_id }),
 *     201,
 *   );
 * ```
 */
export function resolveResponse<
  TSuccess extends ResponseWithData,
  TCode extends number,
>(
  promise: Promise<TSuccess>,
  expected: TCode,
): Promise<ExtractResponseData<Extract<TSuccess, { status: TCode }>>>;
export function resolveResponse<TSuccess extends ResponseWithData>(
  promise: Promise<TSuccess>,
): Promise<ExtractResponseData<SuccessfulResponses<TSuccess>>>;
export async function resolveResponse<
  TSuccess extends ResponseWithData,
  TCode extends number,
>(promise: Promise<TSuccess>, expected?: TCode) {
  const res = await promise;
  const isSuccessfulStatus =
    typeof res.status === "number" && res.status >= 200 && res.status < 300;
