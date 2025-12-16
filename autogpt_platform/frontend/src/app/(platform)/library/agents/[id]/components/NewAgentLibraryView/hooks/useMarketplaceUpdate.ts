import { useGetV2ListMySubmissions } from "@/app/api/__generated__/endpoints/store/store";
import * as React from "react";
import { useState } from "react";

interface UseMarketplaceUpdateProps {
  agent:
    | {
        id: string;
        graph_id: string;
        graph_version: number;
      }
    | null
    | undefined;
}

export function useMarketplaceUpdate({ agent }: UseMarketplaceUpdateProps) {
  const { data: mySubmissions } = useGetV2ListMySubmissions();
  const [modalOpen, setModalOpen] = useState(false);

  // Check if this agent has a newer version than what's published in marketplace
  const hasAgentMarketplaceUpdate = React.useMemo(() => {
    if (!agent || !mySubmissions || mySubmissions.status !== 200) return false;

    const submissionsData = mySubmissions.data.submissions;

    // Find the highest published agent_version for this agent from approved submissions
    // Use graph_id instead of id because submissions store the graph ID
    const publishedVersion = submissionsData
      .filter(
        (s: any) => s.status === "APPROVED" && s.agent_id === agent.graph_id,
      )
      .reduce(
        (max: number | undefined, s: any) =>
          max === undefined || s.agent_version > max ? s.agent_version : max,
        undefined,
      );

    return (
      publishedVersion !== undefined && agent.graph_version > publishedVersion
    );
  }, [agent, mySubmissions]);

  const handlePublishUpdate = () => {
    setModalOpen(true);
  };

  return {
    hasAgentMarketplaceUpdate,
    modalOpen,
    setModalOpen,
    handlePublishUpdate,
  };
}
