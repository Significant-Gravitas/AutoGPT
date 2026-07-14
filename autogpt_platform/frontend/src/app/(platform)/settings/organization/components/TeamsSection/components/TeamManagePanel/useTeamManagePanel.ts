"use client";

import { useState } from "react";

import {
  useDeleteV2RemoveMemberFromWorkspace,
  useGetV2ListWorkspaceMembers,
  usePatchV2UpdateWorkspaceMemberRole,
  usePostV2AddMemberToWorkspace,
  usePostV2LeaveWorkspace,
} from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { TeamMemberResponse } from "@/app/api/__generated__/models/teamMemberResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { TEAM_HEADER_NAME } from "@/services/org-team/headers";

interface Args {
  orgId: string;
  wsId: string;
  currentUserId: string | undefined;
  onChanged: () => void;
  onLeft: () => void;
}

export function useTeamManagePanel({
  orgId,
  wsId,
  currentUserId,
  onChanged,
  onLeft,
}: Args) {
  const [memberToRemove, setMemberToRemove] =
    useState<TeamMemberResponse | null>(null);
  const [isLeaveOpen, setIsLeaveOpen] = useState(false);

  const request = { headers: { [TEAM_HEADER_NAME]: wsId } };

  const membersQuery = useGetV2ListWorkspaceMembers(orgId, wsId, {
    query: {
      enabled: Boolean(orgId && wsId),
      select: (res) => res.data as TeamMemberResponse[],
    },
    request,
  });

  const members = membersQuery.data ?? [];
  const currentTeamMember =
    members.find((m) => m.user_id === currentUserId) ?? null;
  const isTeamAdmin = Boolean(currentTeamMember?.is_admin);

  const { mutateAsync: addMember, isPending: isAdding } =
    usePostV2AddMemberToWorkspace({
      mutation: {
        onError: (error) => {
          toast({
            title: "Failed to add member",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
      request,
    });

  const { mutateAsync: updateRole, isPending: isUpdatingRole } =
    usePatchV2UpdateWorkspaceMemberRole({
      mutation: {
        onError: (error) => {
          toast({
            title: "Failed to update role",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
      request,
    });

  const { mutateAsync: removeMember, isPending: isRemoving } =
    useDeleteV2RemoveMemberFromWorkspace({
      mutation: {
        onError: (error) => {
          toast({
            title: "Failed to remove member",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
      request,
    });

  const { mutateAsync: leaveTeam, isPending: isLeaving } =
    usePostV2LeaveWorkspace({
      mutation: {
        onError: (error) => {
          toast({
            title: "Failed to leave team",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
      request,
    });

  async function handleAddMember(userId: string, label: string) {
    await addMember({ orgId, wsId, data: { user_id: userId } });
    toast({ title: `Added ${label}`, variant: "success" });
    membersQuery.refetch();
    onChanged();
  }

  async function handleRoleChange(member: TeamMemberResponse, role: string) {
    await updateRole({
      orgId,
      wsId,
      uid: member.user_id,
      data: { is_admin: role === "admin" },
    });
    toast({
      title: `${member.name || member.email} is now ${role === "admin" ? "an admin" : "a member"}`,
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
    onLeft();
  }

  return {
    members,
    isLoading: membersQuery.isLoading,
    isTeamAdmin,
    memberToRemove,
    setMemberToRemove,
    isLeaveOpen,
    setIsLeaveOpen,
    isAdding,
    isUpdatingRole,
    isRemoving,
    isLeaving,
    handleAddMember,
    handleRoleChange,
    handleRemoveConfirmed,
    handleLeaveConfirmed,
  };
}
