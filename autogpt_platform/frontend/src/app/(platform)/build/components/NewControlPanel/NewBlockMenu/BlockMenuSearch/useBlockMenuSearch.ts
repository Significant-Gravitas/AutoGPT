import { useCallback, useEffect, useState } from "react";
import { useBlockMenuStore } from "@/app/(platform)/build/stores/blockMenuStore";
import { useAddAgentToBuilder } from "../hooks/useAddAgentToBuilder";
import { getPaginationNextPageNumber } from "@/app/api/helpers";
import {
  getGetV2GetBuilderItemCountsQueryKey,
  getGetV2GetBuilderSuggestionsQueryKey,
} from "@/app/api/__generated__/endpoints/default/default";
import {
  getGetV2ListLibraryAgentsQueryKey,
  usePostV2AddMarketplaceAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import {
  getV2GetSpecificAgent,
  useGetV2BuilderSearchInfinite,
} from "@/app/api/__generated__/endpoints/store/store";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { SearchResponse } from "@/app/api/__generated__/models/searchResponse";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useToast } from "@/components/molecules/Toast/use-toast";
import * as Sentry from "@sentry/nextjs";

export const useBlockMenuSearch = () => {
  const { searchQuery, searchId, setSearchId } = useBlockMenuStore();
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
    data: searchData,
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
    if (!searchData?.pages?.length) {
      return;
    }

    const latestPage = searchData.pages[searchData.pages.length - 1];
    const response = latestPage?.data as SearchResponse;
    if (response?.search_id && response.search_id !== searchId) {
      setSearchId(response.search_id);
    }
  }, [searchData, searchId, setSearchId]);

  useEffect(() => {
    if (searchId && !searchQuery) {
      resetSearchSession();
    }
  }, [resetSearchSession, searchId, searchQuery]);

  const allSearchData =
    searchData?.pages?.flatMap((page) => {
      const response = page.data as SearchResponse;
      return response.items;
    }) ?? [];

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
      addAgentToBuilder(libraryAgent);

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
    allSearchData,
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
