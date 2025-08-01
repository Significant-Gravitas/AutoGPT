"use client";

import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import { useAgentsSection } from "./useAgentsSection";
import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { StoreCard } from "../StoreCard/StoreCard";

export interface Agent {
  slug: string;
  agent_name: string;
  agent_image: string;
  creator: string;
  creator_avatar: string;
  sub_heading: string;
  description: string;
  runs: number;
  rating: number;
}

interface AgentsSectionProps {
  sectionTitle: string;
  agents: StoreAgent[];
  hideAvatars?: boolean;
  margin?: string;
}

export const AgentsSection = ({
  sectionTitle,
  agents: allAgents,
  hideAvatars = false,
  margin = "24px",
}: AgentsSectionProps) => {
  // TODO: Update this when we have pagination and shifts to useAgentsSection
  const displayedAgents = allAgents;
  const { handleCardClick } = useAgentsSection();

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="w-full max-w-[1360px]">
        <h2
          style={{ marginBottom: margin }}
          className="font-poppins text-lg font-semibold text-[#282828] dark:text-neutral-200"
        >
          {sectionTitle}
        </h2>
        {!displayedAgents || displayedAgents.length === 0 ? (
          <div className="text-center text-gray-500 dark:text-gray-400">
            No agents found
          </div>
        ) : (
          <>
            {/* Mobile Carousel View */}
            <Carousel
              className="md:hidden"
              opts={{
                loop: true,
              }}
            >
              <CarouselContent>
                {displayedAgents.map((agent, index) => (
                  <CarouselItem key={index} className="min-w-64 max-w-71">
                    <StoreCard
                      agentName={agent.agent_name}
                      agentImage={agent.agent_image}
                      description={agent.description}
                      runs={agent.runs}
                      rating={agent.rating}
                      avatarSrc={agent.creator_avatar}
                      creatorName={agent.creator}
                      hideAvatar={hideAvatars}
                      onClick={() => handleCardClick(agent.creator, agent.slug)}
                    />
                  </CarouselItem>
                ))}
              </CarouselContent>
            </Carousel>

            <div className="hidden grid-cols-1 place-items-center gap-6 md:grid md:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
              {displayedAgents.map((agent, index) => (
                <StoreCard
                  key={index}
                  agentName={agent.agent_name}
                  agentImage={agent.agent_image}
                  description={agent.description}
                  runs={agent.runs}
                  rating={agent.rating}
                  avatarSrc={agent.creator_avatar}
                  creatorName={agent.creator}
                  hideAvatar={hideAvatars}
                  onClick={() => handleCardClick(agent.creator, agent.slug)}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};
