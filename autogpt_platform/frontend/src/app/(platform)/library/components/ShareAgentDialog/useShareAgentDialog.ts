"use client";

import {
  getGetV2ListGrantsOnAGraphQueryKey,
  useDeleteV2RevokeAGrant,
  useGetV2ListGrantsOnAGraph,
  usePostV2ShareGraphWithATeam,
} from "@/app/api/__generated__/endpoints/grants/grants";
import type { GrantResponse } from "@/app/api/__generated__/models/grantResponse";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useOrgTeamStore } from "@/services/org-team/store";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  CredentialMode,
  GrantCapability,
  PRINCIPAL_TYPE_TEAM,
} from "./helpers";

export function useShareAgentDialog(agent: LibraryAgent, isOpen: boolean) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const orgId = useOrgTeamStore((s) => s.activeOrgID);
  const teams = useOrgTeamStore((s) => s.teams);

  const [teamId, setTeamId] = useState<string | null>(null);
  const [capability, setCapability] = useState<string>(GrantCapability.Execute);
  const [followLatest, setFollowLatest] = useState(false);
  const [credentialMode, setCredentialMode] = useState<string>(
    CredentialMode.Consumer,
  );

  // Only the graph owner may hand out their own credentials; everyone else
  // shares in consumer mode (the team runs with their own connected accounts).
  const isOwner = agent.can_access_graph;

  const grantsQuery = useGetV2ListGrantsOnAGraph(orgId ?? "", agent.graph_id, {
    query: {
      enabled: Boolean(orgId) && isOpen,
      select: (res) => res.data as GrantResponse[],
    },
  });

  const { mutateAsync: shareGraph, isPending: isSharing } =
    usePostV2ShareGraphWithATeam();
  const { mutateAsync: revokeGrant } = useDeleteV2RevokeAGrant();

  function invalidateGrants() {
    if (!orgId) return;
    queryClient.invalidateQueries({
      queryKey: getGetV2ListGrantsOnAGraphQueryKey(orgId, agent.graph_id),
    });
  }

  async function handleShare() {
    if (!orgId || !teamId) return;
    try {
      await shareGraph({
        orgId,
        graphId: agent.graph_id,
        data: {
          principal_type: PRINCIPAL_TYPE_TEAM,
          principal_id: teamId,
          capability,
          // Owner-only choice; omit for non-owners so the backend applies the
          // consumer default rather than trusting a client-sent value.
          credential_mode: isOwner ? credentialMode : undefined,
          follow_latest: followLatest,
          // Pin to the current version unless the sharer opts into "latest".
          graph_version: followLatest ? null : agent.graph_version,
        },
      });
      toast({ title: "Agent shared" });
      setTeamId(null);
      invalidateGrants();
    } catch (error: unknown) {
      toast({
        title: "Failed to share agent",
        description:
          error instanceof Error ? error.message : "Please try again.",
        variant: "destructive",
      });
    }
  }

  async function handleRevoke(grantId: string) {
    if (!orgId) return;
    try {
      await revokeGrant({ orgId, graphId: agent.graph_id, grantId });
      toast({ title: "Access revoked" });
      invalidateGrants();
    } catch (error: unknown) {
      toast({
        title: "Failed to revoke access",
        description:
          error instanceof Error ? error.message : "Please try again.",
        variant: "destructive",
      });
    }
  }

  return {
    teams,
    teamId,
    setTeamId,
    capability,
    setCapability,
    followLatest,
    setFollowLatest,
    credentialMode,
    setCredentialMode,
    isOwner,
    grants: grantsQuery.data ?? [],
    isLoadingGrants: grantsQuery.isLoading,
    isGrantsError: grantsQuery.isError,
    isSharing,
    canShare: Boolean(orgId) && Boolean(teamId),
    handleShare,
    handleRevoke,
  };
}
