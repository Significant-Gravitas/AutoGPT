import * as React from "react";
import {
  useGetV2GetMyAgents,
  useGetV2ListMySubmissions,
} from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import type { MyAgent } from "@/app/api/__generated__/models/myAgent";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";

export interface Agent {
  name: string;
  id: string;
  version: number;
  lastEdited: string;
  imageSrc: string;
  description: string;
  recommendedScheduleCron: string | null;
  isMarketplaceUpdate: boolean; // true if this is an update to existing published agent
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
    publishedSubmissionData?: StoreSubmission | null, // For pre-filling updates
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

  const {
    data: myAgents,
    isLoading: agentsLoading,
    error: agentsError,
  } = useGetV2GetMyAgents();
  const {
    data: mySubmissions,
    isLoading: submissionsLoading,
    error: submissionsError,
  } = useGetV2ListMySubmissions();

  const isLoading = agentsLoading || submissionsLoading;
  const error = agentsError || submissionsError;

  const agents: Agent[] = React.useMemo(() => {
    // Properly handle API responses with okData helper
    const agentsData = (okData(myAgents) as any)?.agents || [];
    const submissionsData = (okData(mySubmissions) as any)?.submissions || [];

    if (agentsData.length === 0) {
      return [];
    }

    return agentsData
      .map((agent: MyAgent): Agent | null => {
        // Find the highest published agent_version for this agent from approved submissions
        const publishedVersion = submissionsData
          .filter(
            (s: StoreSubmission) =>
              s.status === "APPROVED" && s.agent_id === agent.agent_id,
          )
          .reduce(
            (max: number | undefined, s: StoreSubmission) =>
              max === undefined || s.agent_version > max
                ? s.agent_version
                : max,
            undefined,
          );
        const isMarketplaceUpdate =
          publishedVersion !== undefined &&
          agent.agent_version > publishedVersion;
        const isNewAgent = publishedVersion === undefined;

        // Only include agents that are either new or have newer versions than published
        if (!isNewAgent && !isMarketplaceUpdate) {
          return null;
        }

        return {
          name: agent.agent_name,
          id: agent.agent_id,
          version: agent.agent_version,
          lastEdited: agent.last_edited.toLocaleDateString(),
          imageSrc: agent.agent_image || "https://picsum.photos/300/200",
          description: agent.description || "",
          recommendedScheduleCron: agent.recommended_schedule_cron ?? null,
          isMarketplaceUpdate,
        };
      })
      .filter((agent: Agent | null): agent is Agent => agent !== null)
      .sort(
        (a: Agent, b: Agent) =>
          new Date(b.lastEdited).getTime() - new Date(a.lastEdited).getTime(),
      );
  }, [myAgents, mySubmissions]);

  // Function to get published submission data for pre-filling updates
  const getPublishedSubmissionData = (agentId: string) => {
    const submissionsData = (okData(mySubmissions) as any)?.submissions || [];

    const approvedSubmissions = submissionsData
      .filter(
        (submission: StoreSubmission) =>
          submission.agent_id === agentId && submission.status === "APPROVED",
      )
      .sort(
        (a: StoreSubmission, b: StoreSubmission) =>
          b.agent_version - a.agent_version,
      );

    return approvedSubmissions[0] || null;
  };

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
      const selectedAgent = agents.find(
        (agent) => agent.id === selectedAgentId,
      );
      if (selectedAgent) {
        // Get published submission data for pre-filling if this is an update
        const publishedSubmissionData = selectedAgent.isMarketplaceUpdate
          ? getPublishedSubmissionData(selectedAgentId)
          : undefined;

        onNext(
          selectedAgentId,
          selectedAgentVersion,
          {
            name: selectedAgent.name,
            description: selectedAgent.description,
            imageSrc: selectedAgent.imageSrc,
            recommendedScheduleCron: selectedAgent.recommendedScheduleCron,
          },
          publishedSubmissionData,
        );
      }
    }
  };

  // Helper to get published version for an agent
  const getPublishedVersion = (agentId: string): number | undefined => {
    const submissionsData = (okData(mySubmissions) as any)?.submissions || [];

    return submissionsData
      .filter(
        (s: StoreSubmission) =>
          s.status === "APPROVED" && s.agent_id === agentId,
      )
      .reduce(
        (max: number | undefined, s: StoreSubmission) =>
          max === undefined || s.agent_version > max ? s.agent_version : max,
        undefined,
      );
  };

  return {
    // Data
    agents,
    isLoading,
    error,
    // State
    selectedAgentId,
    // Handlers
    handleAgentClick,
    handleNext,
    // Utils
    getPublishedSubmissionData,
    getPublishedVersion,
    // Computed
    isNextDisabled: !selectedAgentId || !selectedAgentVersion,
  };
}
