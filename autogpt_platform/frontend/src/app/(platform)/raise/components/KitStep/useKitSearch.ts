import { useGetV2ListLibraryAgents } from "@/app/api/__generated__/endpoints/library/library";
import { useListCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import {
  useGetV2ListMarketplaceSkills,
  useGetV2ListStoreAgents,
} from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
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
  const isWorkflowScope = scope === "marketplace";
  const hub = useFlagStatus(Flag.SKILLS_HUB);
  const searchHubSkills = !isWorkflowScope && hub.ready && hub.enabled;

  const storeQuery = useGetV2ListStoreAgents(
    { search_query: trimmed, page_size: MAX_SEARCH_RESULTS },
    {
      query: {
        enabled: isWorkflowScope,
        select: (response) => okData(response)?.agents ?? [],
      },
    },
  );
  const libraryQuery = useGetV2ListLibraryAgents(
    { search_term: trimmed, page_size: MAX_SEARCH_RESULTS, is_hidden: false },
    {
      query: {
        enabled: isWorkflowScope,
        select: (response) => okData(response)?.agents ?? [],
      },
    },
  );
  const hubQuery = useGetV2ListMarketplaceSkills(
    { search_query: trimmed, page_size: MAX_SEARCH_RESULTS },
    {
      query: {
        enabled: searchHubSkills,
        select: (response) => okData(response)?.skills ?? [],
      },
    },
  );
  const skillsQuery = useListCopilotSkills(undefined, {
    query: {
      enabled: !isWorkflowScope,
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
      marketplaceSkills: hubQuery.data ?? [],
      scope,
    }),
    isSearching:
      (!isWorkflowScope && skillsQuery.isLoading) ||
      (searchHubSkills && hubQuery.isFetching) ||
      (isWorkflowScope && storeQuery.isFetching) ||
      (isWorkflowScope && libraryQuery.isFetching),
  };
}
