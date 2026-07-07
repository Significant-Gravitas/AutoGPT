"use client";

import {
  useGetV2ListPendingInvitationsForCurrentUser,
  usePostV2AcceptInvitation,
  usePostV2DeclineInvitation,
} from "@/app/api/__generated__/endpoints/invitations/invitations";
import type { UserInvitationResponse } from "@/app/api/__generated__/models/userInvitationResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useOrgTeamStore } from "@/services/org-team/store";

export function useMyInvitationsSection() {
  const { setActiveOrg } = useOrgTeamStore();

  const invitationsQuery = useGetV2ListPendingInvitationsForCurrentUser({
    query: {
      select: (res) => res.data as UserInvitationResponse[],
    },
  });

  const { mutateAsync: acceptInvitation, isPending: isAccepting } =
    usePostV2AcceptInvitation({
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

  const { mutateAsync: declineInvitation, isPending: isDeclining } =
    usePostV2DeclineInvitation({
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
    await acceptInvitation({ token: invitation.token });
    toast({
      title: `Joined ${invitation.org_name}`,
      variant: "success",
    });
    // Switch into the new org — the provider reloads the org list, and
    // resetQueries refetches everything under the new context.
    setActiveOrg(invitation.org_id);
    getQueryClient().resetQueries();
  }

  async function handleDecline(invitation: UserInvitationResponse) {
    await declineInvitation({ token: invitation.token });
    toast({
      title: `Declined invitation from ${invitation.org_name}`,
      variant: "success",
    });
    invitationsQuery.refetch();
  }

  return {
    invitations: invitationsQuery.data ?? [],
    isAccepting,
    isDeclining,
    handleAccept,
    handleDecline,
  };
}
