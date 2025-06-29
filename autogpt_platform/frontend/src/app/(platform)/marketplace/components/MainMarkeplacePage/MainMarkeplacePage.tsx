"use client";
import { Separator } from "@/components/ui/separator";
import { FeaturedSection } from "../FeaturedSection/FeaturedSection";
import { HeroSection } from "../HeroSection/HeroSection";
import { AgentsSection } from "../AgentsSection/AgentsSection";
import { FeaturedCreators } from "../FeaturedCreators/FeaturedCreators";
import { BecomeACreator } from "../BecomeACreator/BecomeACreator";
import {
  useGetV2ListStoreAgents,
  useGetV2ListStoreCreators,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { CreatorsResponse } from "@/app/api/__generated__/models/creatorsResponse";

export const MainMarkeplacePage = () => {
  // Below queries are already fetched on server and hydrated properly in cache, hence these requests are fast
  const {
    data: featuredAgents,
    isLoading: isFeaturedAgentsLoading,
    isError: isFeaturedAgentsError,
  } = useGetV2ListStoreAgents(
    { featured: true },
    {
      query: {
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
    },
    {
      query: {
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

  // TODO : Add better Loading Skeletons
  if (isLoading) {
    return (
      <div className="mx-auto w-screen max-w-[1360px]">
        <main className="px-4">
          <div className="flex min-h-[400px] items-center justify-center">
            <div className="text-lg">Loading...</div>
          </div>
        </main>
      </div>
    );
  }

  // TODO : Add better Error UI
  if (hasError) {
    return (
      <div className="mx-auto w-screen max-w-[1360px]">
        <main className="px-4">
          <div className="flex min-h-[400px] items-center justify-center">
            <div className="text-lg text-red-500">
              Error loading marketplace data. Please try again later.
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="mx-auto w-screen max-w-[1360px]">
      <main className="px-4">
        <HeroSection />
        {featuredAgents && (
          <FeaturedSection featuredAgents={featuredAgents.agents} />
        )}
        {/* 100px margin because our featured sections button are placed 40px below the container */}
        <Separator className="mb-6 mt-24" />

        {topAgents && (
          <AgentsSection sectionTitle="Top Agents" agents={topAgents.agents} />
        )}
        <Separator className="mb-[25px] mt-[60px]" />
        {featuredCreators && (
          <FeaturedCreators featuredCreators={featuredCreators.creators} />
        )}
        <Separator className="mb-[25px] mt-[60px]" />
        <BecomeACreator
          title="Become a Creator"
          description="Join our ever-growing community of hackers and tinkerers"
          buttonText="Become a Creator"
        />
      </main>
    </div>
  );
};
