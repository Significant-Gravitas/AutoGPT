import { useState } from "react";

interface usePublishAgentSelectProps {
  onSelect: (agentId: string, agentVersion: number) => void;
}

export const usePublishAgentSelect = ({
  onSelect,
}: usePublishAgentSelectProps) => {
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [selectedAgentVersion, setSelectedAgentVersion] = useState<
    number | null
  >(null);

  const handleAgentClick = (
    agentName: string,
    agentId: string,
    agentVersion: number,
  ) => {
    setSelectedAgentId(agentId);
    setSelectedAgentVersion(agentVersion);
    onSelect(agentId, agentVersion);
  };

  return {
    selectedAgentId,
    selectedAgentVersion,
    handleAgentClick,
  };
};
