import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useGetSearchGlobalSearch } from "@/app/api/__generated__/endpoints/search/search";
import type { GlobalSearchResponse } from "@/app/api/__generated__/models/globalSearchResponse";
import { buildActionsBucket } from "./actions";
import { buildBucketsFromResponse } from "./helpers";
import { buildNavigationBucket } from "./navigation";

const DEBOUNCE_MS = 200;
const PER_TYPE_LIMIT = 4;

export function useGlobalSearch(isOpen: boolean) {
  const pathname = usePathname();
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");

  useEffect(() => {
    if (!isOpen) {
      setQuery("");
      setDebouncedQuery("");
    }
  }, [isOpen]);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      setDebouncedQuery(query);
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(timeout);
  }, [query]);

  const { data, isFetching, isError } = useGetSearchGlobalSearch(
    { q: debouncedQuery.trim(), per_type_limit: PER_TYPE_LIMIT },
    {
      query: {
        enabled: isOpen,
        // Keep the previous bucket of results visible while a new
        // query is in-flight so the modal doesn't flash empty on every
        // keystroke.
        placeholderData: (prev) => prev,
        // Recent-items (empty query) responses are cached server-side
        // but a short client cache cuts the request on quick re-opens.
        staleTime: 10_000,
      },
    },
  );

  const response =
    data?.status === 200 ? (data.data as GlobalSearchResponse) : undefined;

  const { buckets: searchBuckets, itemsById } =
    buildBucketsFromResponse(response);

  // Client-side command buckets (navigation + actions) filtered against
  // the live query (no debounce — local match is instant). On an exact
  // name match ("Builder", "Copy user ID") the bucket is hoisted above
  // the search results; otherwise it trails them so search stays the
  // primary intent. Priority order on exact match: navigation, actions.
  const commandBuckets = [
    buildNavigationBucket(query, pathname),
    buildActionsBucket(query),
  ];
  const topBuckets = commandBuckets.flatMap((entry) =>
    entry.bucket && entry.isExactMatch ? [entry.bucket] : [],
  );
  const bottomBuckets = commandBuckets.flatMap((entry) =>
    entry.bucket && !entry.isExactMatch ? [entry.bucket] : [],
  );

  const buckets = [...topBuckets, ...searchBuckets, ...bottomBuckets];

  return {
    query,
    setQuery,
    buckets,
    itemsById,
    isFetching,
    isError,
  };
}
