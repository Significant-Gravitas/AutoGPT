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
  margin?: string;
  className?: string;
}

export const AgentsSection: React.FC<AgentsSectionProps> = ({
  sectionTitle,
  agents: allAgents,
  hideAvatars = false,
  className,
}) => {
  const router = useRouter();

  // TODO: Update this when we have pagination
  const displayedAgents = allAgents;

  const handleCardClick = (creator: string, slug: string) => {
    router.push(
      `/marketplace/agent/${encodeURIComponent(creator)}/${encodeURIComponent(slug)}`,
    );
  };

  return (
    <div className={`w-full space-y-9 ${className}`}>
      <h2 className="font-poppins text-base font-medium text-zinc-500">
        {sectionTitle ? sectionTitle : "Top agents"}
      </h2>
      {!displayedAgents || displayedAgents.length === 0 ? (
        <div className="font-poppins text-gray-500 dark:text-gray-400">
          No agents found
        </div>
      ) : (
        <div className="grid grid-cols-1 place-items-center gap-5 md:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
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
      )}
    </div>
  );
};
