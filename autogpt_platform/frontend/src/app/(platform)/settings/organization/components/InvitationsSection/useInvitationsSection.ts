"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

import {
  useDeleteV2RevokeInvitation,
  useGetV2ListPendingInvitations,
  usePostV2CreateInvitation,
} from "@/app/api/__generated__/endpoints/invitations/invitations";
import type { InvitationResponse } from "@/app/api/__generated__/models/invitationResponse";
import { toast } from "@/components/molecules/Toast/use-toast";

const inviteSchema = z.object({
  email: z.string().trim().email("Enter a valid email address"),
  isAdmin: z.boolean(),
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

  const form = useForm<InviteFormValues>({
    resolver: zodResolver(inviteSchema),
    defaultValues: { email: "", isAdmin: false },
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
    await createInvitation({
      orgId,
      data: { email: values.email, is_admin: values.isAdmin },
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
    isLoading: invitationsQuery.isLoading,
    isInviting,
    isRevoking,
    handleInvite,
    handleRevoke,
  };
}
