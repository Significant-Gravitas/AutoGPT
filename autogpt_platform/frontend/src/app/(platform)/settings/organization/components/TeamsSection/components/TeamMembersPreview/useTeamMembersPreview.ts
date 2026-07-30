"use client";

import { useState } from "react";

import {
  useDeleteV2RemoveMemberFromWorkspace,
  useGetV2ListWorkspaceMembers,
  usePatchV2UpdateWorkspaceMemberRole,
  usePostV2LeaveWorkspace,
} from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { TeamMemberResponse } from "@/app/api/__generated__/models/teamMemberResponse";
import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { TEAM_HEADER_NAME } from "@/services/org-team/headers";

interface Args {
  orgId: string;
  team: TeamResponse;
  orgIsAdmin: boolean;
  onChanged: () => void;
}

export function useTeamMembersPreview({
  orgId,
  team,
  orgIsAdmin,
  onChanged,
}: Args) {
  const wsId = team.id;
  const { user } = useAuth();
  const currentUserId = user?.id;

  const [memberToRemove, setMemberToRemove] =
    useState<TeamMemberResponse | null>(null);
  const [isLeaveOpen, setIsLeaveOpen] = useState(false);

  // Same hook (and query key) the manage panel uses, so an already-fetched
  // roster is served from cache. Only mounted once the row is expanded, so the
  // fetch is lazy. The private-team gate (403/404) lands on the team-routes
  // chain; treat it as "you can't inspect this team" rather than a hard error.
  const request = { headers: { [TEAM_HEADER_NAME]: wsId } };
  const membersQuery = useGetV2ListWorkspaceMembers(orgId, wsId, {
    query: {
      enabled: Boolean(orgId && wsId),
      select: (res) => res.data as TeamMemberResponse[],
      retry: false,
    },
    request,
  });

  const members = membersQuery.data ?? [];

  const error: unknown = membersQuery.error;
  const status = error instanceof ApiError ? error.status : undefined;
  const isPrivate = status === 403 || status === 404;

  // A caller manages this roster if they're an admin of this team (their own
  // row's is_admin) or an admin/owner at the org level.
  const callerIsTeamAdmin = Boolean(
    members.find((m) => m.user_id === currentUserId)?.is_admin,
  );
  const canManage = callerIsTeamAdmin || orgIsAdmin;

  const { mutateAsync: updateRole, isPending: isUpdatingRole } =
    usePatchV2UpdateWorkspaceMemberRole({
      mutation: {
        onError: (e) => {
          toast({
            title: "Failed to update role",
            description: e instanceof Error ? e.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
      request,
    });

  const { mutateAsync: removeMember, isPending: isRemoving } =
    useDeleteV2RemoveMemberFromWorkspace({
      mutation: {
        onError: (e) => {
          toast({
            title: "Failed to remove member",
            description: e instanceof Error ? e.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
      request,
    });

  const { mutateAsync: leaveTeam, isPending: isLeaving } =
    usePostV2LeaveWorkspace({
      mutation: {
        onError: (e) => {
          toast({
            title: "Failed to leave team",
            description: e instanceof Error ? e.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
      request,
    });

  async function handleToggleAdmin(member: TeamMemberResponse) {
    const nextAdmin = !member.is_admin;
    await updateRole({
      orgId,
      wsId,
      uid: member.user_id,
      data: { is_admin: nextAdmin },
    });
    toast({
      title: `${member.name || member.email} is now ${nextAdmin ? "an admin" : "a member"}`,
      variant: "success",
    });
    membersQuery.refetch();
  }

  async function handleRemoveConfirmed() {
    if (!memberToRemove) return;
    await removeMember({ orgId, wsId, uid: memberToRemove.user_id });
    toast({
      title: `Removed ${memberToRemove.name || memberToRemove.email}`,
      variant: "success",
    });
    setMemberToRemove(null);
    membersQuery.refetch();
    onChanged();
  }

  async function handleLeaveConfirmed() {
    await leaveTeam({ orgId, wsId });
    toast({ title: "You left the team", variant: "success" });
    setIsLeaveOpen(false);
    membersQuery.refetch();
    onChanged();
  }

  return {
    members,
    isLoading: membersQuery.isLoading,
    isError: membersQuery.isError,
    isPrivate,
    currentUserId,
    canManage,
    memberToRemove,
    setMemberToRemove,
    isLeaveOpen,
    setIsLeaveOpen,
    isUpdatingRole,
    isRemoving,
    isLeaving,
    handleToggleAdmin,
    handleRemoveConfirmed,
    handleLeaveConfirmed,
  };
}
