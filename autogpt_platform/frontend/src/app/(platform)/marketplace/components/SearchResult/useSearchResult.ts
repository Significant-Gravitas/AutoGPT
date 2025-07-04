import {
  useGetV2ListStoreAgents,
  useGetV2ListStoreCreators,
} from "@/app/api/__generated__/endpoints/store/store";
import { CreatorsResponse } from "@/app/api/__generated__/models/creatorsResponse";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { useState, useCallback } from "react";

interface useSearchResultProps {
  sort: string;
  searchTerm: string;
}

export const useSearchResult = ({ searchTerm, sort }: useSearchResultProps) => {
  const [showAgents, setShowAgents] = useState(true);
  const [showCreators, setShowCreators] = useState(true);

  const {
    data: agents,
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
    data: creators,
    isLoading: isCreatorsLoading,
    isError: isCreatorsError,
  } = useGetV2ListStoreCreators(
    { search_query: searchTerm },
    {
      query: {
        select: (x) => {
          return (x.data as CreatorsResponse).creators;
        },
      },
    },
  );

  const isLoading = isCreatorsLoading || isAgentsLoading;
  const isError = isCreatorsError || isAgentsError;

  const agentsCount = agents?.length || 0;
  const creatorsCount = creators?.length || 0;
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

  // TODO: Need to fix it
  const handleSortChange = useCallback(
    (sortValue: string) => {
      let sortBy = "recent";
      if (sortValue === "runs") {
        sortBy = "runs";
      } else if (sortValue === "rating") {
        sortBy = "rating";
      }
      return { sortBy };
    },
    [agents, creators],
  );

  return {
    agents,
    creators,
    isLoading,
    isError,
    agentsCount,
    creatorsCount,
    totalCount,
    showAgents,
    showCreators,
    handleFilterChange,
    handleSortChange,
  };
};
