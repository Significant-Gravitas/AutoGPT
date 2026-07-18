"use client";

import { useState } from "react";

import {
  useDeleteV2RemoveMemberFromOrganization,
  usePatchV2UpdateMemberRole,
} from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import { toast } from "@/components/molecules/Toast/use-toast";

import {
  roleLabel,
  roleToFlags,
  type OrgRole,
} from "../OrgRoleSelect/roleAccess";

interface Args {
  orgId: string;
  onChanged: () => void;
}

export function useMembersSection({ orgId, onChanged }: Args) {
  const [memberToRemove, setMemberToRemove] =
    useState<OrgMemberResponse | null>(null);

  const { mutateAsync: updateRole, isPending: isUpdatingRole } =
    usePatchV2UpdateMemberRole({
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
    });

  const { mutateAsync: removeMember, isPending: isRemoving } =
    useDeleteV2RemoveMemberFromOrganization({
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
    });

  async function handleRoleChange(member: OrgMemberResponse, role: OrgRole) {
    await updateRole({
      orgId,
      uid: member.user_id,
      data: roleToFlags(role),
    });
    toast({
      title: `${member.name || member.email} is now ${roleLabel(role)}`,
      variant: "success",
    });
    onChanged();
  }

  async function handleRemoveConfirmed() {
    if (!memberToRemove) return;
    await removeMember({ orgId, uid: memberToRemove.user_id });
    toast({
      title: `Removed ${memberToRemove.name || memberToRemove.email}`,
      variant: "success",
    });
    setMemberToRemove(null);
    onChanged();
  }

  return {
    memberToRemove,
    setMemberToRemove,
    isUpdatingRole,
    isRemoving,
    handleRoleChange,
    handleRemoveConfirmed,
  };
}
