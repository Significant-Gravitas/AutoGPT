import { useGetV2ListStoreAgents } from "@/app/api/__generated__/endpoints/store/store";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { useState } from "react";

const queryConfig = {
  staleTime: 60 * 1000, // 60 seconds - match server cache
  gcTime: 5 * 60 * 1000, // 5 minutes
  refetchOnWindowFocus: false, // Avoid unnecessary refetches
  refetchOnMount: false, // Use cached data from server
};

export const useMainMarketplacePage = () => {
  const [category, setCategory] = useState<string | null>(null);

  // Data is prefetched on server and hydrated, these queries will use cached data
  const {
    data: featuredAgents,
    isLoading: isFeaturedAgentsLoading,
    isError: isFeaturedAgentsError,
  } = useGetV2ListStoreAgents(
    { featured: true },
    {
      query: {
        ...queryConfig,
        select: (x) => {
          return x.data as StoreAgentsResponse;
        },
      },
    },
  );

  const {
    data: topAgents,
    isLoading: isTopAgentsLoading,
    isError: isTopAgentsError,
  } = useGetV2ListStoreAgents(
    {
      sorted_by: "runs",
      page_size: 1000,
      ...(category ? { category } : {}),
    },
    {
      query: {
        ...queryConfig,
        // Keep the current grid on screen while a category change loads, so
        // picking a filter doesn't collapse the whole page to skeletons.
        placeholderData: (previousData) => previousData,
        select: (x) => {
          return x.data as StoreAgentsResponse;
        },
      },
    },
  );

  const isLoading = isFeaturedAgentsLoading || isTopAgentsLoading;
  const hasError = isFeaturedAgentsError || isTopAgentsError;

  return {
    featuredAgents,
    topAgents,
    category,
    setCategory,
    isLoading,
    hasError,
  };
};
