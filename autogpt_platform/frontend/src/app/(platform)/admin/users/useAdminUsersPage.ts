"use client";

import type { BulkInvitedUsersResponse } from "@/app/api/__generated__/models/bulkInvitedUsersResponse";
import { okData } from "@/app/api/helpers";
import {
  getGetV2ListInvitedUsersQueryKey,
  useGetV2ListInvitedUsers,
  usePostV2BulkCreateInvitedUsers,
  usePostV2CreateInvitedUser,
  usePostV2RetryInvitedUserTally,
  usePostV2RevokeInvitedUser,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";

function getErrorMessage(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }

  return "Something went wrong";
}

export function useAdminUsersPage() {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [bulkInviteFile, setBulkInviteFile] = useState<File | null>(null);
  const [bulkInviteInputKey, setBulkInviteInputKey] = useState(0);
  const [lastBulkInviteResult, setLastBulkInviteResult] =
    useState<BulkInvitedUsersResponse | null>(null);
  const [pendingInviteAction, setPendingInviteAction] = useState<string | null>(
    null,
  );

  const invitedUsersQuery = useGetV2ListInvitedUsers({
    query: {
      select: okData,
      refetchInterval: 30_000,
    },
  });

  const createInvitedUserMutation = usePostV2CreateInvitedUser({
    mutation: {
      onSuccess: async () => {
        setEmail("");
        setName("");
        await queryClient.invalidateQueries({
          queryKey: getGetV2ListInvitedUsersQueryKey(),
        });
        toast({
          title: "Invited user created",
          variant: "default",
        });
      },
      onError: (error) => {
        toast({
          title: getErrorMessage(error),
          variant: "destructive",
        });
      },
    },
  });

  const bulkCreateInvitedUsersMutation = usePostV2BulkCreateInvitedUsers({
    mutation: {
      onSuccess: async (response) => {
        const result = okData(response) ?? null;
        setBulkInviteFile(null);
        setBulkInviteInputKey((currentValue) => currentValue + 1);
        setLastBulkInviteResult(result);
        await queryClient.invalidateQueries({
          queryKey: getGetV2ListInvitedUsersQueryKey(),
        });
        toast({
          title: result
            ? `${result.created_count} invites created`
            : "Bulk invite upload complete",
          variant: "default",
        });
      },
      onError: (error) => {
        toast({
          title: getErrorMessage(error),
          variant: "destructive",
        });
      },
    },
  });

  const retryInvitedUserTallyMutation = usePostV2RetryInvitedUserTally({
    mutation: {
      onSuccess: async () => {
        setPendingInviteAction(null);
        await queryClient.invalidateQueries({
          queryKey: getGetV2ListInvitedUsersQueryKey(),
        });
        toast({
          title: "Tally pre-seeding restarted",
          variant: "default",
        });
      },
      onError: (error) => {
        setPendingInviteAction(null);
        toast({
          title: getErrorMessage(error),
          variant: "destructive",
        });
      },
    },
  });

  const revokeInvitedUserMutation = usePostV2RevokeInvitedUser({
    mutation: {
      onSuccess: async () => {
        setPendingInviteAction(null);
        await queryClient.invalidateQueries({
          queryKey: getGetV2ListInvitedUsersQueryKey(),
        });
        toast({
          title: "Invite revoked",
          variant: "default",
        });
      },
      onError: (error) => {
        setPendingInviteAction(null);
        toast({
          title: getErrorMessage(error),
          variant: "destructive",
        });
      },
    },
  });

  function handleCreateInvite(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    createInvitedUserMutation.mutate({
      data: {
        email,
        name: name.trim() || null,
      },
    });
  }

  function handleRetryTally(invitedUserId: string) {
    setPendingInviteAction(`retry:${invitedUserId}`);
    retryInvitedUserTallyMutation.mutate({ invitedUserId });
  }

  function handleBulkInviteFileChange(file: File | null) {
    setBulkInviteFile(file);
  }

  function handleBulkInviteSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!bulkInviteFile) {
      return;
    }

    bulkCreateInvitedUsersMutation.mutate({
      data: {
        file: bulkInviteFile,
      },
    });
  }

  function handleRevoke(invitedUserId: string) {
    setPendingInviteAction(`revoke:${invitedUserId}`);
    revokeInvitedUserMutation.mutate({ invitedUserId });
  }

  return {
    email,
    name,
    bulkInviteFile,
    bulkInviteInputKey,
    lastBulkInviteResult,
    invitedUsers: invitedUsersQuery.data?.invited_users ?? [],
    invitedUsersError: invitedUsersQuery.error,
    isLoadingInvitedUsers: invitedUsersQuery.isLoading,
    isRefreshingInvitedUsers: invitedUsersQuery.isFetching,
    isCreatingInvite: createInvitedUserMutation.isPending,
    isBulkInviting: bulkCreateInvitedUsersMutation.isPending,
    pendingInviteAction,
    setEmail,
    setName,
    handleBulkInviteFileChange,
    handleBulkInviteSubmit,
    handleCreateInvite,
    handleRetryTally,
    handleRevoke,
  };
}
