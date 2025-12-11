import type { InfiniteData } from "@tanstack/react-query";
import {
  getV1IsOnboardingEnabled,
  getV1OnboardingState,
} from "./__generated__/endpoints/onboarding/onboarding";
import { Pagination } from "./__generated__/models/pagination";

/**
 * Narrow an orval response to its success payload if and only if it is a `200` status with OK shape.
 *
 * Usage with React Query select:
 * ```ts
 *   const { data: agent } = useGetV2GetLibraryAgent(agentId, {
 *     query: { select: okData },
 *   });
 *
 *   data // is now properly typed as LibraryAgent | undefined
 * ```
 */
export function okData<TResponse extends { status: number; data?: object }>(
  res: TResponse | undefined,
): (TResponse & { status: 200 })["data"] | undefined {
  if (!res || typeof res !== "object") return undefined;

  // status must exist and be exactly 200
  const maybeStatus = (res as { status?: unknown }).status;
  if (maybeStatus !== 200) return undefined;

  // data must exist and be an object/array/string/number/etc. We only need to
  // check presence to safely return it as T; the generic T is enforced at call sites.
  if (!("data" in (res as Record<string, unknown>))) return undefined;

  return res.data;
}

export function getPaginatedTotalCount(
  infiniteData: InfiniteData<unknown> | undefined,
  fallbackCount?: number,
): number {
  const lastPage = infiniteData?.pages.at(-1);
  if (!hasValidPaginationInfo(lastPage)) return fallbackCount ?? 0;
  return lastPage.data.pagination.total_items ?? fallbackCount ?? 0;
}

export function getPaginationNextPageNumber(
  lastPage:
    | { data: { pagination?: Pagination; [key: string]: any } }
    | undefined,
): number | undefined {
  if (!hasValidPaginationInfo(lastPage)) return undefined;

  const { pagination } = lastPage.data;
  const hasMore =
    pagination.current_page * pagination.page_size < pagination.total_items;
  return hasMore ? pagination.current_page + 1 : undefined;
}

export function unpaginate<
  TItemData extends object,
  TPageDataKey extends string,
>(
  infiniteData: InfiniteData<{
    status: number;
    data: { [key in TPageDataKey]: TItemData[] } | Record<string, any>;
  }>,
  pageListKey: TPageDataKey &
    keyof (typeof infiniteData)["pages"][number]["data"],
): TItemData[] {
  return (
    infiniteData?.pages.flatMap((page) => {
      if (!hasValidListPage<TItemData, TPageDataKey>(page, pageListKey))
        return [];
      return page.data[pageListKey] || [];
    }) || []
  );
}

function hasValidListPage<TItemData extends object, TKey extends string>(
  page: unknown,
  pageListKey: TKey,
): page is { data: { [key in TKey]: TItemData[] } } {
  return (
    typeof page === "object" &&
    page !== null &&
    "data" in page &&
    typeof page.data === "object" &&
    page.data !== null &&
    pageListKey in page.data &&
    Array.isArray((page.data as Record<string, unknown>)[pageListKey])
  );
}

function hasValidPaginationInfo(
  page: unknown,
): page is { data: { pagination: Pagination; [key: string]: any } } {
  return (
    typeof page === "object" &&
    page !== null &&
    "data" in page &&
    typeof page.data === "object" &&
    page.data !== null &&
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

  if (typeof expected === "number") {
    if (res.status !== expected) {
      throw new Error(`Unexpected status ${res.status}`);
    }
  } else if (!isSuccessfulStatus) {
    throw new Error(`Unexpected status ${res.status}`);
  }

  return res.data;
}

export async function shouldShowOnboarding() {
  const isEnabled = await resolveResponse(getV1IsOnboardingEnabled());
  const onboarding = await resolveResponse(getV1OnboardingState());
  const isCompleted = onboarding.completedSteps.includes("CONGRATS");
  return isEnabled && !isCompleted;
}
