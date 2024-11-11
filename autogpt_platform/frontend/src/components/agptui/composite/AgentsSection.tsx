"use client";

import * as React from "react";
import { StoreCard } from "@/components/agptui/StoreCard";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import { useRouter } from "next/navigation";
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
  agents: Agent[];
  hideAvatars?: boolean;
}

export const AgentsSection: React.FC<AgentsSectionProps> = ({
  sectionTitle,
  agents: topAgents,
  hideAvatars = false,
}) => {
  const router = useRouter();

  const handleCardClick = (creator: string, slug: string) => {
    router.push(`/store/${creator}/${slug}`);
  };

  return (
    <div className="flex flex-col items-center justify-center py-4 lg:py-8">
      <div className="w-full">
        <div className="mb-6 font-neue text-[23px] font-bold leading-9 tracking-tight text-[#282828]">
          {sectionTitle}
        </div>
        <Carousel
          className="md:hidden"
          opts={{
            loop: true,
          }}
        >
          <CarouselContent>
            {topAgents.map((agent, index) => (
              <CarouselItem key={index} className="min-w-64 max-w-68">
                <StoreCard
                  agentName={agent.agent_name}
                  agentImage={agent.agent_image}
                  description={agent.description}
                  runs={agent.runs}
                  rating={agent.rating}
                  avatarSrc={agent.creator_avatar}
                  hideAvatar={hideAvatars}
                  onClick={() => handleCardClick(agent.creator, agent.slug)}
                />
              </CarouselItem>
            ))}
          </CarouselContent>
        </Carousel>
        <div className="hidden grid-cols-1 place-items-center gap-3 md:grid md:grid-cols-2 lg:grid-cols-3">
          {topAgents.map((agent, index) => (
            <StoreCard
              key={index}
              agentName={agent.agent_name}
              agentImage={agent.agent_image}
              description={agent.description}
              runs={agent.runs}
              rating={agent.rating}
              avatarSrc={agent.creator_avatar}
              hideAvatar={hideAvatars}
              onClick={() => handleCardClick(agent.creator, agent.slug)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
