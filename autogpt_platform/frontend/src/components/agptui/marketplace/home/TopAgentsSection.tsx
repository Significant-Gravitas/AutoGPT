import * as React from "react";
import { StoreCard } from "../../StoreCard";

interface TopAgent {
  agentName: string;
  description: string;
  runs: number;
  rating: number;
}

interface TopAgentsSectionProps {
  topAgents: TopAgent[];
  onCardClick: (agentName: string) => void;
}

export const TopAgentsSection: React.FC<TopAgentsSectionProps> = ({
  topAgents,
  onCardClick,
}) => {
  return (
    <div className="flex flex-col items-center justify-center py-8">
      <div className="w-full">
        <div className="mb-6 font-neue text-[23px] font-bold leading-9 tracking-tight text-[#282828]">
          Top agents
        </div>
        <div className="grid w-full grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {topAgents.map((agent, index) => (
            <StoreCard
              key={index}
              agentName={agent.agentName}
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
