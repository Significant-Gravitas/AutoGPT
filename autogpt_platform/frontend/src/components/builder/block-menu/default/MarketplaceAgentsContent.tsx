import React, { useState, useEffect } from "react";
import MarketplaceAgentBlock from "../MarketplaceAgentBlock";
import { marketplaceAgentData } from "../../testing_data";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { StoreAgent } from "@/lib/autogpt-server-api";

const MarketplaceAgentsContent: React.FC = () => {
  const [agents, setAgents] = useState<StoreAgent[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const api = useBackendAPI();

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const response = await api.getStoreAgents();
        // BLOCK MENU TODO : figure out how to add agent in flow and add pagination as well
        setAgents(response.agents);
        setLoading(false);
      } catch (err) {
        setLoading(false);
      }
    };

    fetchAgents();
  }, [api]);

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
    <div className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
      <div className="w-full space-y-3 px-4 pb-4">
        {agents.map((agent) => (
          <MarketplaceAgentBlock
            key={agent.slug}
            title={agent.agent_name}
            image_url={agent.agent_image}
            creator_name={agent.creator}
            number_of_runs={agent.runs}
          />
        ))}
      </div>
    </div>
  );
};

export default MarketplaceAgentsContent;
