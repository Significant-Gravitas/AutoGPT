import * as React from "react";
import { StoreCard } from "../../StoreCard";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";

interface TopAgent {
  agentName: string;
  agentImage: string;
  description: string;
  runs: number;
  rating: number;
  avatarSrc: string; // Added avatarSrc to match StoreCard props
}

interface TopAgentsSectionProps {
  sectionTitle: string;
  topAgents: TopAgent[];
  onCardClick: (agentName: string) => void;
}

export const TopAgentsSection: React.FC<TopAgentsSectionProps> = ({
  sectionTitle,
  topAgents,
  onCardClick,
}) => {
  return (
    <div className="flex flex-col items-center justify-center py-8">
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
                  agentName={agent.agentName}
                  agentImage={agent.agentImage}
                  description={agent.description}
                  runs={agent.runs}
                  rating={agent.rating}
                  avatarSrc={agent.avatarSrc}
                  onClick={() => onCardClick(agent.agentName)}
                />
              </CarouselItem>
            ))}
          </CarouselContent>
        </Carousel>
        <div className="hidden grid-cols-1 place-items-center gap-3 md:grid md:grid-cols-2 lg:grid-cols-3">
          {topAgents.map((agent, index) => (
            <StoreCard
              key={index}
              agentName={agent.agentName}
              agentImage={agent.agentImage}
              description={agent.description}
              runs={agent.runs}
              rating={agent.rating}
              avatarSrc={agent.avatarSrc}
              onClick={() => onCardClick(agent.agentName)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
