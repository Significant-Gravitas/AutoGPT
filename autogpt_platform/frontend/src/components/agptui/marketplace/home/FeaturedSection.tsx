import * as React from "react";
import { FeaturedStoreCard } from "../../FeaturedStoreCard";

interface FeaturedAgent {
  agentName: string;
  agentImage: string;
  creatorName: string;
  description: string;
  runs: number;
  rating: number;
}

interface FeaturedSectionProps {
  featuredAgents: FeaturedAgent[];
  onCardClick: (agentName: string) => void;
}

export const FeaturedSection: React.FC<FeaturedSectionProps> = ({
  featuredAgents,
  onCardClick,
}) => {
  return (
    <div className="flex flex-col items-center justify-center px-4 py-8">
      <div className="w-full px-4 lg:px-16">
        <h2 className="mb-6 font-neue text-2xl font-bold leading-tight tracking-tight text-[#282828] sm:mb-8 sm:text-3xl">
          Featured agents
        </h2>

        <div className="flex flex-col items-center justify-center gap-6 sm:flex-row sm:flex-wrap lg:flex-nowrap">
          {featuredAgents.slice(0, 2).map((agent, index) => (
            <FeaturedStoreCard
              key={index}
              agentName={agent.agentName}
              agentImage={agent.agentImage}
              creatorName={agent.creatorName}
              description={agent.description}
              runs={agent.runs}
              rating={agent.rating}
              onClick={() => onCardClick(agent.agentName)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
