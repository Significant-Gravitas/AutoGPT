import {
  getGetV2ListLibraryAgentsQueryKey,
  useGetV2ListLibraryAgents,
} from "@/app/api/__generated__/endpoints/library/library";
import { getListCopilotSkillsQueryKey } from "@/app/api/__generated__/endpoints/skills/skills";
import { useListCopilotSkills } from "@/services/skills/scoped-queries";
import {
  getGetV2ListStoreAgentsQueryKey,
  useGetV2ListMarketplaceSkills,
  useGetV2ListStoreAgents,
} from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { useState } from "react";
import {
  combineSearchHits,
  MAX_SEARCH_RESULTS,
  SEARCH_DEBOUNCE_MS,
  type KitSearchScope,
} from "./helpers";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";

export function useKitSearch(scope: KitSearchScope) {
  const organizationId = useOrgTeamStore((state) => state.activeOrgID);
  const teamId = useOrgTeamStore((state) => state.activeTeamID);
  const isTenantReady = useOrgTeamStore((state) => state.isLoaded);
  const [searchQuery, setSearchQuery] = useState("");
  const debouncedQuery = useDebouncedValue(searchQuery, SEARCH_DEBOUNCE_MS);
  const trimmed = debouncedQuery.trim();
  const hasQuery = trimmed.length > 0;
  const isWorkflowScope = scope === "marketplace";

  const searchHubSkills = !isWorkflowScope;

  const hubQuery = useGetV2ListMarketplaceSkills(
    { search_query: trimmed, page_size: MAX_SEARCH_RESULTS },
    {
      query: {
        enabled: searchHubSkills,
        select: (response) => okData(response)?.skills ?? [],
      },
    },
  );
  const storeParams = {
    search_query: trimmed,
    page_size: MAX_SEARCH_RESULTS,
  };
  const storeQuery = useGetV2ListStoreAgents(storeParams, {
    query: {
      enabled: isWorkflowScope && isTenantReady,
      queryKey: getTeamScopedQueryKey(
        getGetV2ListStoreAgentsQueryKey(storeParams),
        organizationId,
        teamId,
      ),
      select: (response) => okData(response)?.agents ?? [],
    },
    request: getTenantRequestInit(organizationId, teamId, isTenantReady),
  });
  const libraryParams = {
    search_term: trimmed,
    page_size: MAX_SEARCH_RESULTS,
    is_hidden: false,
  };
  const libraryQuery = useGetV2ListLibraryAgents(libraryParams, {
    query: {
      enabled: isWorkflowScope && isTenantReady,
      queryKey: getTeamScopedQueryKey(
        getGetV2ListLibraryAgentsQueryKey(libraryParams),
        organizationId,
        teamId,
      ),
      select: (response) => okData(response)?.agents ?? [],
    },
    request: getTenantRequestInit(organizationId, teamId, isTenantReady),
  });
  const skillsQuery = useListCopilotSkills(undefined, {
    request: getTenantRequestInit(organizationId, teamId, isTenantReady),
    query: {
      enabled: !isWorkflowScope && isTenantReady,
      queryKey: getTeamScopedQueryKey(
        getListCopilotSkillsQueryKey(),
        organizationId,
        teamId,
      ),
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
