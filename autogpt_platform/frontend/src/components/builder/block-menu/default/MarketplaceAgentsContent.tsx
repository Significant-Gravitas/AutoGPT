import React, { useState, useEffect } from "react";
import MarketplaceAgentBlock from "../MarketplaceAgentBlock";
import { marketplaceAgentData } from "../../testing_data";

export interface MarketplaceAgent {
  id: number;
  title: string;
  image_url: string;
  creator_name: string;
  number_of_runs: number;
}

const MarketplaceAgentsContent: React.FC = () => {
  const [agents, setAgents] = useState<MarketplaceAgent[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  // Update Block Menu fetching
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        setAgents(marketplaceAgentData);
        setLoading(false);
      } catch (err) {
        setLoading(false);
      }
    };

    fetchAgents();
  }, []);

  if (loading) {
    return (
      <div className="w-full space-y-3 p-4">
        {Array(5)
          .fill(null)
          .map((_, index) => (
            <MarketplaceAgentBlock.Skeleton key={index} />
          ))}
      </div>
    );
  }

  return (
    <div className="scrollbar-thumb-rounded scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200 h-full overflow-y-scroll pt-4">
      <div className="w-full space-y-3 px-4 pb-4">
        {agents.map((agent) => (
          <MarketplaceAgentBlock
            key={agent.id}
            title={agent.title}
            image_url={agent.image_url}
            creator_name={agent.creator_name}
            number_of_runs={agent.number_of_runs}
          />
        ))}
      </div>
    </div>
  );
};

export default MarketplaceAgentsContent;
