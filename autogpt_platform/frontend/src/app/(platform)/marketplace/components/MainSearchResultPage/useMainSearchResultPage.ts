import {
  useGetV2ListStoreAgents,
  useGetV2ListStoreCreators,
} from "@/app/api/__generated__/endpoints/store/store";
import { CreatorsResponse } from "@/app/api/__generated__/models/creatorsResponse";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { useState, useMemo } from "react";

interface useMainSearchResultPageType {
  searchTerm: string;
  sort: string;
}

export const useMainSearchResultPage = ({
  searchTerm,
  sort,
}: useMainSearchResultPageType) => {
  const [showAgents, setShowAgents] = useState(true);
  const [showCreators, setShowCreators] = useState(true);
  const [clientSortBy, setClientSortBy] = useState<string>(sort);

  const {
    data: agentsData,
    isLoading: isAgentsLoading,
    isError: isAgentsError,
  } = useGetV2ListStoreAgents(
    {
      search_query: searchTerm,
      sorted_by: sort,
    },
    {
      query: {
        select: (x) => {
          return (x.data as StoreAgentsResponse).agents;
        },
      },
    },
  );

  const {
    data: creatorsData,
    isLoading: isCreatorsLoading,
    isError: isCreatorsError,
  } = useGetV2ListStoreCreators(
    { search_query: searchTerm, sorted_by: sort },
    {
      query: {
        select: (x) => {
          return (x.data as CreatorsResponse).creators;
        },
      },
    },
  );

  // This is the strategy, we are using for sorting the agents and creators.
  // currently we are doing it client side but maybe we will shift it to the server side.
  // we will store the sortBy state in the url params, and then refetch the data with the new sortBy.

  const agents = useMemo(() => {
    if (!agentsData) return [];

    const sorted = [...agentsData];

    if (clientSortBy === "runs") {
      return sorted.sort((a, b) => b.runs - a.runs);
    } else if (clientSortBy === "rating") {
      return sorted.sort((a, b) => b.rating - a.rating);
    } else {
      return sorted;
    }
  }, [agentsData, clientSortBy]);

  const creators = useMemo(() => {
    if (!creatorsData) return [];

    const sorted = [...creatorsData];

    if (clientSortBy === "runs") {
      return sorted.sort((a, b) => b.agent_runs - a.agent_runs);
    } else if (clientSortBy === "rating") {
      return sorted.sort((a, b) => b.agent_rating - a.agent_rating);
    } else {
      return sorted.sort((a, b) => b.num_agents - a.num_agents);
    }
  }, [creatorsData, clientSortBy]);

  const agentsCount = agents?.length ?? 0;
  const creatorsCount = creators?.length ?? 0;
  const totalCount = agentsCount + creatorsCount;

  const handleFilterChange = (value: string) => {
    if (value === "agents") {
      setShowAgents(true);
      setShowCreators(false);
    } else if (value === "creators") {
      setShowAgents(false);
      setShowCreators(true);
    } else {
      setShowAgents(true);
      setShowCreators(true);
    }
  };

  const handleSortChange = (sortValue: string) => {
    setClientSortBy(sortValue);
  };

  return {
    agents,
    creators,
    handleFilterChange,
    handleSortChange,
    agentsCount,
    creatorsCount,
    totalCount,
    showAgents,
    showCreators,
    isAgentsLoading,
    isCreatorsLoading,
    isAgentsError,
    isCreatorsError,
  };
};
