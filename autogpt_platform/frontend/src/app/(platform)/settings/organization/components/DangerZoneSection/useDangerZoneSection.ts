"use client";

import { useState } from "react";

import {
  useDeleteV2DeleteOrganization,
  usePostV2TransferOrganizationOwnership,
} from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgMemberResponse } from "@/app/api/__generated__/models/orgMemberResponse";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useOrgTeamStore } from "@/services/org-team/store";

interface Args {
  org: OrgResponse;
  members: OrgMemberResponse[];
  onTransferred: () => void;
}

export function useDangerZoneSection({ org, members, onTransferred }: Args) {
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [transferTargetId, setTransferTargetId] = useState<string | null>(null);
  const [isTransferConfirmOpen, setIsTransferConfirmOpen] = useState(false);
  const { orgs, setOrgs, setActiveOrg } = useOrgTeamStore();

  const transferableMembers = members.filter((member) => !member.is_owner);
  const transferTarget =
    transferableMembers.find((member) => member.user_id === transferTargetId) ??
    null;

  const { mutateAsync: deleteOrg, isPending: isDeleting } =
    useDeleteV2DeleteOrganization({
      mutation: {
        onError: (error) => {
          toast({
            title: "Failed to delete organization",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  const { mutateAsync: transferOwnership, isPending: isTransferring } =
    usePostV2TransferOrganizationOwnership({
      mutation: {
        onError: (error) => {
          toast({
            title: "Failed to transfer ownership",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  async function handleDeleteConfirmed() {
    try {
      await deleteOrg({ orgId: org.id });
    } catch {
      return;
    }
    const remaining = orgs.filter((o) => o.id !== org.id);
    setOrgs(remaining);
    const personal = remaining.find((o) => o.isPersonal) ?? remaining[0];
    if (personal) {
      setActiveOrg(personal.id);
    }
    getQueryClient().resetQueries();
    toast({ title: `Organization "${org.name}" deleted`, variant: "success" });
    setIsDeleteOpen(false);
  }

  async function handleTransferConfirmed() {
    if (!transferTarget) return;
    try {
      await transferOwnership({
        orgId: org.id,
        data: { new_owner_id: transferTarget.user_id },
      });
    } catch {
      return;
    }
    toast({
      title: `${transferTarget.name || transferTarget.email} is now the owner of ${org.name}`,
      variant: "success",
    });
    setIsTransferConfirmOpen(false);
    setTransferTargetId(null);
    onTransferred();
  }

  return {
    isDeleteOpen,
    setIsDeleteOpen,
    isDeleting,
    handleDeleteConfirmed,
    transferableMembers,
    transferTarget,
    transferTargetId,
    setTransferTargetId,
    isTransferConfirmOpen,
    setIsTransferConfirmOpen,
    isTransferring,
    handleTransferConfirmed,
  };
}
