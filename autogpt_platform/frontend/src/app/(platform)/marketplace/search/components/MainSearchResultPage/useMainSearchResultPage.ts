import {
  useGetV2ListMarketplaceSkills,
  useGetV2ListStoreAgents,
  useGetV2ListStoreCreators,
} from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { CreatorsResponse } from "@/app/api/__generated__/models/creatorsResponse";
import { GetV2ListStoreAgentsParams } from "@/app/api/__generated__/models/getV2ListStoreAgentsParams";
import { GetV2ListStoreCreatorsParams } from "@/app/api/__generated__/models/getV2ListStoreCreatorsParams";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import {
  Flag,
  useFlagStatus,
  useGetFlag,
} from "@/services/feature-flags/use-get-flag";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useExpertsSection } from "../../../components/ExpertsSection/useExpertsSection";
import { useState, useMemo } from "react";

type MarketplaceSearchSort = GetV2ListStoreAgentsParams["sorted_by"];
type CreatorSortBy = GetV2ListStoreCreatorsParams["sorted_by"];

interface useMainSearchResultPageType {
  searchTerm: string;
  sort: MarketplaceSearchSort;
}

export const useMainSearchResultPage = ({
  searchTerm,
  sort,
}: useMainSearchResultPageType) => {
  const [showAgents, setShowAgents] = useState(true);
  const [showCreators, setShowCreators] = useState(true);
  const [showSkills, setShowSkills] = useState(true);
  const [showExperts, setShowExperts] = useState(true);
  const skillsHub = useFlagStatus(Flag.SKILLS_HUB);
  const { isLoggedIn, isUserLoading } = useAuth();
  const isHireExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  // Same gate as the marketplace shelf: expert pages are public, hiring is
  // not, so a signed-in user outside the beta sees neither surface.
  const isExpertsVisible =
    !isUserLoading && (!isLoggedIn || isHireExpertsEnabled);
  const [clientSortBy, setClientSortBy] = useState<string>(
    sort ?? "updated_at",
  );

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

  const creatorsSortBy: CreatorSortBy = useMemo(() => {
    switch (sort) {
      case "runs":
        return "agent_runs";
      case "rating":
        return "agent_rating";
      default:
        return "num_agents";
    }
  }, [sort]);
  const {
    data: creatorsData,
    isLoading: isCreatorsLoading,
    isError: isCreatorsError,
  } = useGetV2ListStoreCreators(
    {
      search_query: searchTerm,
      sorted_by: creatorsSortBy,
    },
    {
      query: {
        select: (x) => {
          return (x.data as CreatorsResponse).creators;
        },
      },
    },
  );

  const { data: skillsData, isLoading: isSkillsLoading } =
    useGetV2ListMarketplaceSkills(
      { search_query: searchTerm },
      {
        query: {
          enabled: skillsHub.ready && skillsHub.enabled,
          select: (response) => okData(response)?.skills ?? [],
        },
      },
    );

  const {
    templates: experts,
    hiredTemplateIds,
    isLoading: isExpertsLoading,
  } = useExpertsSection({
    searchQuery: searchTerm,
    enabled: isExpertsVisible,
  });

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

  const skills = skillsData ?? [];
  const agentsCount = agents?.length ?? 0;
  const creatorsCount = creators?.length ?? 0;
  const skillsCount = skills.length;
  const expertsCount = isExpertsVisible ? experts.length : 0;
  const totalCount = agentsCount + creatorsCount + skillsCount + expertsCount;

  const handleFilterChange = (value: string) => {
    setShowAgents(value === "all" || value === "agents");
    setShowCreators(value === "all" || value === "creators");
    setShowSkills(value === "all" || value === "skills");
    setShowExperts(value === "all" || value === "experts");
  };

  const handleSortChange = (sortValue: string) => {
    setClientSortBy(sortValue);
  };

  return {
    agents,
    creators,
    skills,
    experts,
    hiredTemplateIds,
    handleFilterChange,
    handleSortChange,
    agentsCount,
    creatorsCount,
    skillsCount,
    expertsCount,
    totalCount,
    showAgents,
    showCreators,
    showSkills,
    showExperts,
    isExpertsVisible,
    isSkillsHubEnabled: skillsHub.ready && skillsHub.enabled,
    isAgentsLoading,
    isCreatorsLoading,
    isSkillsLoading,
    isExpertsLoading: isExpertsVisible && isExpertsLoading,
    isAgentsError,
    isCreatorsError,
  };
};
