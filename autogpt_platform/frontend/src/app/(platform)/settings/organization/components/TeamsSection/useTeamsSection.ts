"use client";

import { useState } from "react";

import {
  useDeleteV2DeleteWorkspace,
  useGetV2ListWorkspaces,
  usePostV2SelfJoinOpenWorkspace,
} from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";
import { toast } from "@/components/molecules/Toast/use-toast";

interface Args {
  orgId: string;
}

export function useTeamsSection({ orgId }: Args) {
  const [expandedTeamId, setExpandedTeamId] = useState<string | null>(null);
  const [teamToDelete, setTeamToDelete] = useState<TeamResponse | null>(null);
  const [isCreateOpen, setIsCreateOpen] = useState(false);

  const teamsQuery = useGetV2ListWorkspaces(orgId, {
    query: {
      enabled: Boolean(orgId),
      select: (res) => res.data as TeamResponse[],
    },
  });

  const teams = teamsQuery.data ?? [];

  const { mutateAsync: joinTeam, isPending: isJoining } =
    usePostV2SelfJoinOpenWorkspace({
      mutation: {
        onError: (error) => {
          toast({
            title: "Couldn't join team",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  const { mutateAsync: deleteTeam, isPending: isDeleting } =
    useDeleteV2DeleteWorkspace({
      mutation: {
        onError: (error) => {
          toast({
            title: "Failed to delete team",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  async function handleJoin(team: TeamResponse) {
    await joinTeam({ orgId, wsId: team.id });
    toast({ title: `Joined ${team.name}`, variant: "success" });
    teamsQuery.refetch();
  }

  async function handleDeleteConfirmed() {
    if (!teamToDelete) return;
    await deleteTeam({ orgId, wsId: teamToDelete.id });
    toast({ title: `Deleted ${teamToDelete.name}`, variant: "success" });
    if (expandedTeamId === teamToDelete.id) {
      setExpandedTeamId(null);
    }
    setTeamToDelete(null);
    teamsQuery.refetch();
  }

  function toggleExpanded(teamId: string) {
    setExpandedTeamId((current) => (current === teamId ? null : teamId));
  }

  return {
    teams,
    isLoading: teamsQuery.isLoading,
    isError: teamsQuery.isError,
    refetch: teamsQuery.refetch,
    expandedTeamId,
    toggleExpanded,
    setExpandedTeamId,
    teamToDelete,
    setTeamToDelete,
    isJoining,
    isDeleting,
    handleJoin,
    handleDeleteConfirmed,
    isCreateOpen,
    setIsCreateOpen,
  };
}
