import { useState } from "react";
import { AgentTableRowProps } from "../AgentTableRow/AgentTableRow";

interface useAgentTableProps {
  agents: Omit<
    AgentTableRowProps,
    | "setSelectedAgents"
    | "selectedAgents"
    | "onEditSubmission"
    | "onDeleteSubmission"
  >[];
}

export const useAgentTable = ({ agents }: useAgentTableProps) => {
  const [selectedAgents, setSelectedAgents] = useState<Set<string>>(new Set());

  const handleSelectAll = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.checked) {
      setSelectedAgents(new Set(agents.map((agent) => agent.agent_id)));
    } else {
      setSelectedAgents(new Set());
    }
  };
  return { selectedAgents, handleSelectAll, setSelectedAgents };
};
