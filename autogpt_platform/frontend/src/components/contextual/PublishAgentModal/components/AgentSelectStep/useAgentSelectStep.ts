import * as React from "react";
import { useGetV2GetMyAgents } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

export interface Agent {
  name: string;
  id: string;
  version: number;
  lastEdited: string;
  imageSrc: string;
  description: string;
  recommendedScheduleCron: string | null;
}

interface UseAgentSelectStepProps {
  onSelect: (agentId: string, agentVersion: number) => void;
  onNext: (
    agentId: string,
    agentVersion: number,
    agentData: {
      name: string;
      description: string;
      imageSrc: string;
      recommendedScheduleCron: string | null;
    },
  ) => void;
}

export function useAgentSelectStep({
  onSelect,
  onNext,
}: UseAgentSelectStepProps) {
  const [selectedAgentId, setSelectedAgentId] = React.useState<string | null>(
    null,
  );
  const [selectedAgentVersion, setSelectedAgentVersion] = React.useState<
    number | null
  >(null);
  const { isLoggedIn } = useSupabase();

  const {
    data: _myAgents,
    isLoading,
    error,
  } = useGetV2GetMyAgents(undefined, {
    query: {
      enabled: isLoggedIn,
      select: (res) =>
        okData(res)
          ?.agents.map(
            (agent): Agent => ({
              name: agent.agent_name,
              id: agent.agent_id,
              version: agent.agent_version,
              lastEdited: agent.last_edited.toLocaleDateString(),
              imageSrc: agent.agent_image || "https://picsum.photos/300/200",
              description: agent.description || "",
              recommendedScheduleCron: agent.recommended_schedule_cron ?? null,
            }),
          )
          .sort(
            (a: Agent, b: Agent) =>
              new Date(b.lastEdited).getTime() -
              new Date(a.lastEdited).getTime(),
          ),
    },
  });
  const myAgents = _myAgents ?? [];

  const handleAgentClick = (
    _: string,
    agentId: string,
    agentVersion: number,
  ) => {
    setSelectedAgentId(agentId);
    setSelectedAgentVersion(agentVersion);
    onSelect(agentId, agentVersion);
  };

  const handleNext = () => {
    if (selectedAgentId && selectedAgentVersion) {
      const selectedAgent = myAgents.find(
        (agent) => agent.id === selectedAgentId,
      );
      if (selectedAgent) {
        onNext(selectedAgentId, selectedAgentVersion, {
          name: selectedAgent.name,
          description: selectedAgent.description,
          imageSrc: selectedAgent.imageSrc,
          recommendedScheduleCron: selectedAgent.recommendedScheduleCron,
        });
      }
    }
  };

  return {
    // Data
    myAgents,
    isLoading,
    error,
    // State
    selectedAgentId,
    // Handlers
    handleAgentClick,
    handleNext,
    // Computed
    isNextDisabled: !selectedAgentId || !selectedAgentVersion,
  };
}
