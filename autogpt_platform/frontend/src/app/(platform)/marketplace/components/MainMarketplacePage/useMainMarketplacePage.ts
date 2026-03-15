import {
  useGetV2ListStoreAgents,
  useGetV2ListStoreCreators,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { CreatorsResponse } from "@/app/api/__generated__/models/creatorsResponse";

const queryConfig = {
  staleTime: 60 * 1000, // 60 seconds - match server cache
  gcTime: 5 * 60 * 1000, // 5 minutes
  refetchOnWindowFocus: false, // Avoid unnecessary refetches
  refetchOnMount: false, // Use cached data from server
};

export const useMainMarketplacePage = () => {
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
    },
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
    data: featuredCreators,
    isLoading: isFeaturedCreatorsLoading,
    isError: isFeaturedCreatorsError,
  } = useGetV2ListStoreCreators(
    { featured: true, sorted_by: "num_agents" },
    {
      query: {
        ...queryConfig,
        select: (x) => {
          return x.data as CreatorsResponse;
        },
      },
    },
  );

  const isLoading =
    isFeaturedAgentsLoading || isTopAgentsLoading || isFeaturedCreatorsLoading;
  const hasError =
    isFeaturedAgentsError || isTopAgentsError || isFeaturedCreatorsError;

  return {
    featuredAgents,
    topAgents,
    featuredCreators,
    isLoading,
    hasError,
  };
};
