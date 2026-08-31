"use client";

import { useState } from "react";

import {
  useGetV2ListPendingInvitationsForCurrentUser,
  usePostV2AcceptInvitation,
  usePostV2DeclineInvitation,
} from "@/app/api/__generated__/endpoints/invitations/invitations";
import type { UserInvitationResponse } from "@/app/api/__generated__/models/userInvitationResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useOrgTeamStore } from "@/services/org-team/store";

import { getOrgsAfterJoin } from "./helpers";

export function useMyInvitationsSection() {
  const { setOrgs, setActiveOrg } = useOrgTeamStore();
  const [acceptingId, setAcceptingId] = useState<string | null>(null);
  const [decliningId, setDecliningId] = useState<string | null>(null);

  const invitationsQuery = useGetV2ListPendingInvitationsForCurrentUser({
    query: {
      select: (res) => res.data as UserInvitationResponse[],
    },
  });

  const { mutateAsync: acceptInvitation } = usePostV2AcceptInvitation({
    mutation: {
      onError: (error) => {
        toast({
          title: "Failed to accept invitation",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  const { mutateAsync: declineInvitation } = usePostV2DeclineInvitation({
    mutation: {
      onError: (error) => {
        toast({
          title: "Failed to decline invitation",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  async function handleAccept(invitation: UserInvitationResponse) {
    setAcceptingId(invitation.id);
    try {
      await acceptInvitation({ token: invitation.token });
      toast({
        title: `Joined ${invitation.org_name}`,
        variant: "success",
      });
      // Land the joined org in the store before switching into it, then let
      // resetQueries refetch everything under the new context.
      setOrgs(
        await getOrgsAfterJoin(invitation, useOrgTeamStore.getState().orgs),
      );
      setActiveOrg(invitation.org_id);
      getQueryClient().resetQueries();
    } catch {
      // onError already surfaced the failure toast; swallow the rejection so
      // it doesn't escape the click handler unhandled.
    } finally {
      setAcceptingId(null);
    }
  }

  async function handleDecline(invitation: UserInvitationResponse) {
    setDecliningId(invitation.id);
    try {
      await declineInvitation({ token: invitation.token });
      toast({
        title: `Declined invitation from ${invitation.org_name}`,
        variant: "success",
      });
      invitationsQuery.refetch();
    } catch {
      // onError already surfaced the failure toast.
    } finally {
      setDecliningId(null);
    }
  }

  return {
    invitations: invitationsQuery.data ?? [],
    acceptingId,
    decliningId,
    handleAccept,
    handleDecline,
  };
}
