import * as React from "react";
import { keepPreviousData } from "@tanstack/react-query";
import { useGetV2GetMyAgents } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import { MyAgentsSortBy } from "@/app/api/__generated__/models/myAgentsSortBy";
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

const PAGE_SIZE = 10;

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
  // Capture the selected agent's full payload at click time so pagination
  // doesn't drop it off the current page before the user hits Continue.
  const [selectedAgentSnapshot, setSelectedAgentSnapshot] =
    React.useState<Agent | null>(null);
  const [page, setPage] = React.useState(1);
  const [pageDirection, setPageDirection] = React.useState<1 | -1>(1);
  const [sortBy, setSortBy] = React.useState<MyAgentsSortBy>(
    MyAgentsSortBy.most_recent,
  );
  const { isLoggedIn } = useSupabase();

  React.useEffect(() => {
    setPage(1);
  }, [sortBy]);

  const {
    data: agentsData,
    isLoading,
    isFetching,
    error,
  } = useGetV2GetMyAgents(
    {
      page,
      page_size: PAGE_SIZE,
      sort_by: sortBy,
    },
    {
      query: {
        enabled: isLoggedIn,
        refetchOnMount: "always",
        staleTime: 0,
        placeholderData: keepPreviousData,
        select: (res) => {
          const payload = okData(res);
          if (!payload) return null;
          const agents = payload.agents.map(
            (agent): Agent => ({
              name: agent.agent_name,
              id: agent.graph_id,
              version: agent.graph_version,
              lastEdited: agent.last_edited.toLocaleDateString(),
              imageSrc: agent.agent_image || "https://picsum.photos/300/200",
              description: agent.description || "",
              recommendedScheduleCron: agent.recommended_schedule_cron ?? null,
            }),
          );
          return { agents, pagination: payload.pagination };
        },
      },
    },
  );

  const myAgents = agentsData?.agents ?? [];
  const totalPages = agentsData?.pagination?.total_pages ?? 0;
  const totalItems = agentsData?.pagination?.total_items ?? 0;

  function handleAgentClick(_: string, agentId: string, agentVersion: number) {
    const clicked = myAgents.find((a) => a.id === agentId) ?? null;
    setSelectedAgentId(agentId);
    setSelectedAgentVersion(agentVersion);
    setSelectedAgentSnapshot(clicked);
    onSelect(agentId, agentVersion);
  }

  function handleNext() {
    if (!selectedAgentId || !selectedAgentVersion) return;
    const selectedAgent =
      myAgents.find((a) => a.id === selectedAgentId) ?? selectedAgentSnapshot;
    if (!selectedAgent) return;
    onNext(selectedAgentId, selectedAgentVersion, {
      name: selectedAgent.name,
      description: selectedAgent.description,
      imageSrc: selectedAgent.imageSrc,
      recommendedScheduleCron: selectedAgent.recommendedScheduleCron,
    });
  }

  function handleSortChange(value: string) {
    setSortBy(value as MyAgentsSortBy);
  }

  function goToPage(next: number) {
    if (next < 1 || (totalPages > 0 && next > totalPages)) return;
    setPageDirection(next > page ? 1 : -1);
    setPage(next);
  }

  return {
    myAgents,
    isLoading,
    isFetching,
    error,
    selectedAgentId,
    page,
    totalPages,
    totalItems,
    pageSize: PAGE_SIZE,
    sortBy,
    pageDirection,
    handleAgentClick,
    handleNext,
    handleSortChange,
    goToPage,
    isNextDisabled: !selectedAgentId || !selectedAgentVersion,
  };
}
