import { getGetV2GetBuilderItemCountsQueryKey } from "@/app/api/__generated__/endpoints/default/default";
import {
  getGetV2ListLibraryAgentsQueryKey,
  usePostV2AddMarketplaceAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import {
  getV2GetSpecificAgent,
  useGetV2ListStoreAgentsInfinite,
} from "@/app/api/__generated__/endpoints/store/store";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { StoreAgentsResponse } from "@/lib/autogpt-server-api";
import { getQueryClient } from "@/lib/react-query/queryClient";
import * as Sentry from "@sentry/nextjs";
import { useState } from "react";

export const useMarketplaceAgentsContent = () => {
  const { toast } = useToast();
  const [addingAgent, setAddingAgent] = useState<string | null>(null);

  const {
    data: listStoreAgents,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading: isListStoreAgentsLoading,
    isError: isListStoreAgentsError,
    error: listStoreAgentsError,
    refetch: refetchListStoreAgents,
  } = useGetV2ListStoreAgentsInfinite(
    {
      page: 1,
      page_size: 10,
    },
    {
      query: {
        getNextPageParam: (lastPage) => {
          const pagination = (lastPage.data as StoreAgentsResponse).pagination;
          const isMore =
            pagination.current_page * pagination.page_size <
            pagination.total_items;

          return isMore ? pagination.current_page + 1 : undefined;
        },
      },
    },
  );

  const allAgents =
    listStoreAgents?.pages?.flatMap((page) => {
      const response = page.data as StoreAgentsResponse;
      return response.agents;
    }) ?? [];

  const status = listStoreAgents?.pages[0]?.status;

  const { mutate: addMarketplaceAgent } = usePostV2AddMarketplaceAgent({
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
    },
  });

  const handleAddStoreAgent = async ({
    creator_name,
    slug,
  }: {
    creator_name: string;
    slug: string;
  }) => {
    try {
      setAddingAgent(slug);
      const { data: agent, status } = await getV2GetSpecificAgent(
        creator_name,
        slug,
      );
      if (status !== 200) {
        Sentry.captureException("Store listing version not found");
        throw new Error("Store listing version not found");
      }

      addMarketplaceAgent({
        data: {
          store_listing_version_id: agent?.store_listing_version_id,
        },
      });

      // Need a way to convert the library agent into block
      // then add the block in builder
    } catch (error) {
      Sentry.captureException(error);
      toast({
        title: "Error",
        description: "Failed to add agent to library",
      });
    } finally {
      setAddingAgent(null);
    }
  };

  return {
    handleAddStoreAgent,
    listStoreAgents: allAgents,
    status,
    addingAgent,
    isListStoreAgentsLoading,
    isListStoreAgentsError,
    listStoreAgentsError,
    refetchListStoreAgents,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  };
};
