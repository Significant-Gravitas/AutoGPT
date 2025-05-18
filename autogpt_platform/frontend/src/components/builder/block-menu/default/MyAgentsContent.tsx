import React, { useState, useEffect } from "react";
import UGCAgentBlock from "../UGCAgentBlock";
import { myAgentData } from "../../testing_data";

export interface UserAgent {
  id: number;
  title: string;
  edited_time: string;
  version: number;
  image_url: string;
}

const MyAgentsContent: React.FC = () => {
  const [agents, setAgents] = useState<UserAgent[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  // Update Block Menu fetching
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        setAgents(myAgentData);
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
            <UGCAgentBlock.Skeleton key={index} />
          ))}
      </div>
    );
  }

  return (
    <div className="scrollbar-thumb-rounded h-full overflow-y-scroll pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
      <div className="w-full space-y-3 px-4 pb-4">
        {agents.map((agent) => (
          <UGCAgentBlock
            key={agent.id}
            title={agent.title}
            edited_time={agent.edited_time}
            version={agent.version}
            image_url={agent.image_url}
          />
        ))}
      </div>
    </div>
  );
};

export default MyAgentsContent;
