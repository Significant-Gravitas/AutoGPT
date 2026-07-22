"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

import {
  useDeleteV2RevokeInvitation,
  useGetV2ListPendingInvitations,
  usePostV2CreateInvitation,
} from "@/app/api/__generated__/endpoints/invitations/invitations";
import { useGetV2ListWorkspaces } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { InvitationResponse } from "@/app/api/__generated__/models/invitationResponse";
import type { TeamResponse } from "@/app/api/__generated__/models/teamResponse";
import { toast } from "@/components/molecules/Toast/use-toast";

import { ORG_ROLE_VALUES, roleToFlags } from "../OrgRoleSelect/roleAccess";

const inviteSchema = z.object({
  email: z.string().trim().email("Enter a valid email address"),
  role: z.enum(ORG_ROLE_VALUES),
  teamIds: z.array(z.string()),
});

export type InviteFormValues = z.infer<typeof inviteSchema>;

interface Args {
  orgId: string;
  isAdmin: boolean;
}

export function useInvitationsSection({ orgId, isAdmin }: Args) {
  const invitationsQuery = useGetV2ListPendingInvitations(orgId, {
    query: {
      enabled: isAdmin,
      select: (res) => res.data as InvitationResponse[],
    },
  });

  const teamsQuery = useGetV2ListWorkspaces(orgId, {
    query: {
      enabled: isAdmin && Boolean(orgId),
      select: (res) => res.data as TeamResponse[],
    },
  });

  const orgTeams = teamsQuery.data ?? [];

  // Everyone lands in the org's default team automatically on accept
  // (add_org_member), so pre-assignment only makes sense for the other teams.
  const assignableTeams = orgTeams.filter((team) => !team.is_default);

  // Lets pending-invitation rows spell out assigned team names client-side.
  const teamNameById = new Map(orgTeams.map((team) => [team.id, team.name]));

  const form = useForm<InviteFormValues>({
    resolver: zodResolver(inviteSchema),
    defaultValues: { email: "", role: "member", teamIds: [] },
    mode: "onChange",
  });

  const { mutateAsync: createInvitation, isPending: isInviting } =
    usePostV2CreateInvitation({
      mutation: {
        onError: (error) => {
          toast({
            title: "Failed to send invitation",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  const { mutateAsync: revokeInvitation, isPending: isRevoking } =
    useDeleteV2RevokeInvitation({
      mutation: {
        onError: (error) => {
          toast({
            title: "Failed to revoke invitation",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  async function handleInvite(values: InviteFormValues) {
    const flags = roleToFlags(values.role);
    await createInvitation({
      orgId,
      data: {
        email: values.email,
        is_admin: flags.is_admin,
        is_billing_manager: flags.is_billing_manager,
        team_ids: values.teamIds,
      },
    });
    toast({ title: `Invitation sent to ${values.email}`, variant: "success" });
    form.reset();
    invitationsQuery.refetch();
  }

  async function handleRevoke(invitation: InvitationResponse) {
    await revokeInvitation({ orgId, invitationId: invitation.id });
    toast({
      title: `Invitation to ${invitation.email} revoked`,
      variant: "success",
    });
    invitationsQuery.refetch();
  }

  return {
    form,
    invitations: invitationsQuery.data ?? [],
    assignableTeams,
    teamNameById,
    isLoading: invitationsQuery.isLoading,
    isInviting,
    isRevoking,
    handleInvite,
    handleRevoke,
  };
}
