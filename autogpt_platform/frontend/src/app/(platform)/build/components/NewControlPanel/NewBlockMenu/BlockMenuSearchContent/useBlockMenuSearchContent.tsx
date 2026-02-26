import { useCallback, useEffect, useState } from "react";
import { useBlockMenuStore } from "@/app/(platform)/build/stores/blockMenuStore";
import { useAddAgentToBuilder } from "../hooks/useAddAgentToBuilder";
import {
  getPaginationNextPageNumber,
  okData,
  unpaginate,
} from "@/app/api/helpers";
import {
  getGetV2GetBuilderItemCountsQueryKey,
  getGetV2GetBuilderSuggestionsQueryKey,
} from "@/app/api/__generated__/endpoints/default/default";
import {
  getGetV2ListLibraryAgentsQueryKey,
  getV2GetLibraryAgent,
  usePostV2AddMarketplaceAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import {
  getV2GetSpecificAgent,
  useGetV2BuilderSearchInfinite,
} from "@/app/api/__generated__/endpoints/store/store";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useToast } from "@/components/molecules/Toast/use-toast";
import * as Sentry from "@sentry/nextjs";
import { CategoryCounts } from "../BlockMenuFilters/types";

export const useBlockMenuSearchContent = () => {
  const {
    searchQuery,
    searchId,
    setSearchId,
    filters,
    setCreatorsList,
    creators,
    setCategoryCounts,
  } = useBlockMenuStore();

  const { toast } = useToast();
  const { addAgentToBuilder, addLibraryAgentToBuilder } =
    useAddAgentToBuilder();
  const queryClient = getQueryClient();

  const resetSearchSession = useCallback(() => {
    setSearchId(undefined);
    queryClient.invalidateQueries({
      queryKey: getGetV2GetBuilderSuggestionsQueryKey(),
    });
  }, [queryClient, setSearchId]);

  const [addingLibraryAgentId, setAddingLibraryAgentId] = useState<
    string | null
  >(null);
  const [addingMarketplaceAgentSlug, setAddingMarketplaceAgentSlug] = useState<
    string | null
  >(null);

  const {
    data: searchQueryData,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: searchLoading,
  } = useGetV2BuilderSearchInfinite(
    {
      page: 1,
      page_size: 8,
      search_query: searchQuery,
      search_id: searchId,
      filter: filters.length > 0 ? filters.join(",") : undefined,
      by_creator: creators.length > 0 ? creators : undefined,
    },
    {
      query: { getNextPageParam: getPaginationNextPageNumber },
    },
  );

  const { mutateAsync: addMarketplaceAgent } = usePostV2AddMarketplaceAgent({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryAgentsQueryKey(),
        });

        queryClient.refetchQueries({
          queryKey: getGetV2GetBuilderItemCountsQueryKey(),
        });
      },
      onError: (error) => {
        Sentry.captureException(error);
        toast({
          title: "Failed to add agent to library",
          description:
            ((error as any).message as string) ||
            "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  useEffect(() => {
    if (!searchQueryData?.pages?.length) {
      return;
    }

    const lastPage = okData(searchQueryData.pages.at(-1));
    if (lastPage?.search_id && lastPage.search_id !== searchId) {
      setSearchId(lastPage.search_id);
    }
  }, [searchQueryData, searchId, setSearchId]);

  // from all the results, we need to get all the unique creators
  useEffect(() => {
    if (!searchQueryData?.pages?.length) {
      return;
    }
    const latestData = okData(searchQueryData.pages.at(-1));
    setCategoryCounts(
      (latestData?.total_items as CategoryCounts) || {
        blocks: 0,
        integrations: 0,
        marketplace_agents: 0,
        my_agents: 0,
      },
    );
    setCreatorsList(latestData?.items || []);
  }, [searchQueryData]);

  useEffect(() => {
    if (searchId && !searchQuery) {
      resetSearchSession();
    }
  }, [resetSearchSession, searchId, searchQuery]);

  const searchResults = searchQueryData
    ? unpaginate(searchQueryData, "items")
    : [];

  const handleAddLibraryAgent = async (agent: LibraryAgent) => {
    setAddingLibraryAgentId(agent.id);
    try {
      await addLibraryAgentToBuilder(agent);
    } catch (error) {
      console.error("Error adding library agent:", error);
    } finally {
      setAddingLibraryAgentId(null);
    }
  };

  const handleAddMarketplaceAgent = async ({
    creator_name,
    slug,
  }: {
    creator_name: string;
    slug: string;
  }) => {
    try {
      setAddingMarketplaceAgentSlug(slug);
      const { data: agent, status } = await getV2GetSpecificAgent(
        creator_name,
        slug,
      );
      if (status !== 200) {
        Sentry.captureException("Store listing version not found");
        throw new Error("Store listing version not found");
      }

      const response = await addMarketplaceAgent({
        data: {
          store_listing_version_id: agent?.store_listing_version_id,
        },
      });

      const libraryAgent = response.data as LibraryAgent;

      const { data: libraryAgentDetails } = await getV2GetLibraryAgent(
        libraryAgent.id,
      );

      addAgentToBuilder(libraryAgentDetails as LibraryAgent);

      toast({
        title: "Agent Added",
        description: "Agent has been added to your library and builder",
      });
    } catch (error) {
      Sentry.captureException(error);
      toast({
        title: "Failed to add agent to library",
        description:
          ((error as any).message as string) || "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      setAddingMarketplaceAgentSlug(null);
    }
  };

  return {
    searchResults,
    isFetchingNextPage,
    fetchNextPage,
    hasNextPage,
    searchLoading,
    handleAddLibraryAgent,
    handleAddMarketplaceAgent,
    addingLibraryAgentId,
    addingMarketplaceAgentSlug,
  };
};
