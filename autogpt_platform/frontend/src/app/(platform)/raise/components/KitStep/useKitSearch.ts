import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";
import { useListCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import { useGetV2ListStoreAgents } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { useState } from "react";
import {
  combineSearchHits,
  MAX_SEARCH_RESULTS,
  SEARCH_DEBOUNCE_MS,
  type KitSearchScope,
} from "./helpers";

export function useKitSearch(scope: KitSearchScope) {
  const [searchQuery, setSearchQuery] = useState("");
  const debouncedQuery = useDebouncedValue(searchQuery, SEARCH_DEBOUNCE_MS);
  const trimmed = debouncedQuery.trim();
  const hasQuery = trimmed.length > 0;
  const searchStore = scope === "marketplace" || hasQuery;
  const searchLibrary = scope === "marketplace";

  const storeQuery = useGetV2ListStoreAgents(
    { search_query: trimmed, page_size: MAX_SEARCH_RESULTS },
    {
      query: {
        enabled: searchStore,
        select: (response) => okData(response)?.agents ?? [],
      },
    },
  );
  const libraryQuery = useGetV2ListLibraryAgents(
    { search_term: trimmed, page_size: MAX_SEARCH_RESULTS, is_hidden: false },
    {
      query: {
        enabled: searchLibrary,
        select: (response) => okData(response)?.agents ?? [],
      },
    },
  );
  const skillsQuery = useListCopilotSkills({
    query: {
      enabled: scope === "skills",
      select: (response) => okData(response) ?? [],
    },
  });

  return {
    searchQuery,
    setSearchQuery,
    hasQuery,
    hits: combineSearchHits({
      query: trimmed,
      storeAgents: storeQuery.data ?? [],
      libraryAgents: libraryQuery.data ?? [],
      skills: skillsQuery.data ?? [],
      scope,
    }),
    isSearching:
      (scope === "skills" && skillsQuery.isLoading) ||
      (searchStore && storeQuery.isFetching) ||
      (searchLibrary && libraryQuery.isFetching),
  };
}
