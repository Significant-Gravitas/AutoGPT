import { useBlockMenuStore } from "../../../../stores/blockMenuStore";
import { useGetV2BuilderSearchInfinite } from "@/app/api/__generated__/endpoints/store/store";
import { SearchResponse } from "@/app/api/__generated__/models/searchResponse";
import { useState } from "react";
import { useAddAgentToBuilder } from "../hooks/useAddAgentToBuilder";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { getV2GetSpecificAgent } from "@/app/api/__generated__/endpoints/store/store";
import {
  getGetV2ListLibraryAgentsQueryKey,
  usePostV2AddMarketplaceAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import { getGetV2GetBuilderItemCountsQueryKey } from "@/app/api/__generated__/endpoints/default/default";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useToast } from "@/components/molecules/Toast/use-toast";
import * as Sentry from "@sentry/nextjs";

export const useBlockMenuSearch = () => {
  const { searchQuery } = useBlockMenuStore();
  const { toast } = useToast();
  const { addAgentToBuilder, addLibraryAgentToBuilder } =
    useAddAgentToBuilder();

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
    },
    {
      query: {
        getNextPageParam: (lastPage, allPages) => {
          const pagination = lastPage.data as SearchResponse;
          const isMore = pagination.more_pages;
          return isMore ? allPages.length + 1 : undefined;
        },
      },
    },
  );

  const { mutateAsync: addMarketplaceAgent } = usePostV2AddMarketplaceAgent({
    mutation: {
      onSuccess: () => {
        const queryClient = getQueryClient();
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
