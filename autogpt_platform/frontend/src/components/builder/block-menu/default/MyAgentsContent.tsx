import React, { useState, useEffect } from "react";
import UGCAgentBlock from "../UGCAgentBlock";
import { myAgentData } from "../../testing_data";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { LibraryAgent } from "@/lib/autogpt-server-api";

const MyAgentsContent: React.FC = () => {
  const [agents, setAgents] = useState<LibraryAgent[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const api = useBackendAPI();
  // TEMPORARY FETCHING
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        // BLOCK MENU TODO : figure out how to add agent in flow and add pagination as well
        const response = await api.listLibraryAgents();
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
            <UGCAgentBlock.Skeleton key={index} />
          ))}
      </div>
    );
  }

  return (
    <div className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
      <div className="w-full space-y-3 px-4 pb-4">
        {agents.map((agent) => (
          <UGCAgentBlock
            key={agent.id}
            title={agent.name}
            edited_time={agent.updated_at}
            version={agent.graph_version}
            image_url={agent.image_url}
          />
        ))}
      </div>
    </div>
  );
};

export default MyAgentsContent;
