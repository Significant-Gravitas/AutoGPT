import { useEffect, useState } from "react";
import { useGetV2GlobalSearchHybridOnQueryRecentOnEmpty } from "@/app/api/__generated__/endpoints/search/search";
import type { GlobalSearchResponse } from "@/app/api/__generated__/models/globalSearchResponse";
import { buildBucketsFromResponse } from "./helpers";

const DEBOUNCE_MS = 200;
const PER_TYPE_LIMIT = 4;

export function useGlobalSearch(isOpen: boolean) {
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

  const { data, isFetching, isError } =
    useGetV2GlobalSearchHybridOnQueryRecentOnEmpty(
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

  const { buckets, itemsById } = buildBucketsFromResponse(response);

  return {
    query,
    setQuery,
    buckets,
    itemsById,
    isFetching,
    isError,
  };
}
