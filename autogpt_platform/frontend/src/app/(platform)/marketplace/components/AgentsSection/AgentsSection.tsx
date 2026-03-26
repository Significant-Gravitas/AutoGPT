"use client";

import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/__legacy__/ui/carousel";
import { Text } from "@/components/atoms/Text/Text";
import { DotsNineIcon } from "@phosphor-icons/react";
import { StoreCard } from "../StoreCard/StoreCard";
import { useAgentsSection } from "./useAgentsSection";

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

interface Props {
  sectionTitle?: string;
  agents: StoreAgent[];
  hideAvatars?: boolean;
}

export function AgentsSection({
  sectionTitle,
  agents: allAgents,
  hideAvatars = false,
}: Props) {
  const displayedAgents = allAgents;
  const { handleCardClick } = useAgentsSection();

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="w-full max-w-[1360px]">
        {sectionTitle ? (
          <div className="mb-8 flex flex-row items-center gap-2">
            <DotsNineIcon size={24} />
            <Text variant="h4">{sectionTitle}</Text>
          </div>
        ) : null}
        {!displayedAgents || displayedAgents.length === 0 ? (
          <Text variant="body" className="ml-4 mt-8 text-gray-500">
            No agents found
          </Text>
        ) : (
          <>
            {/* Mobile Carousel View */}
            <Carousel
              className="-mx-4 md:hidden"
              opts={{
                loop: true,
              }}
            >
              <div className="relative">
                <CarouselContent className="px-4 pb-2">
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
                        creatorSlug={agent.creator}
                        agentSlug={agent.slug}
                        agentGraphID={agent.agent_graph_id}
                        onClick={() =>
                          handleCardClick(agent.creator, agent.slug)
                        }
                      />
                    </CarouselItem>
                  ))}
                </CarouselContent>
                <div className="pointer-events-none absolute inset-y-0 left-0 w-8 bg-gradient-to-r from-[rgb(246,247,248)] to-transparent" />
                <div className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-[rgb(246,247,248)] to-transparent" />
              </div>
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
                  creatorSlug={agent.creator}
                  agentSlug={agent.slug}
                  agentGraphID={agent.agent_graph_id}
                  onClick={() => handleCardClick(agent.creator, agent.slug)}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
