"use client";

import type { AdminCopilotUserSummary } from "@/app/api/__generated__/models/adminCopilotUserSummary";
import { ChatSessionStartType } from "@/app/api/__generated__/models/chatSessionStartType";
import type { SendCopilotEmailsResponse } from "@/app/api/__generated__/models/sendCopilotEmailsResponse";
import type { TriggerCopilotSessionResponse } from "@/app/api/__generated__/models/triggerCopilotSessionResponse";
import { okData } from "@/app/api/helpers";
import { customMutator } from "@/app/api/mutators/custom-mutator";
import {
  useGetV2SearchCopilotUsers,
  usePostV2TriggerCopilotSession,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useMutation } from "@tanstack/react-query";
import { useDeferredValue, useState } from "react";

function getErrorMessage(error: unknown) {
  if (error instanceof ApiError) {
    if (
      typeof error.response === "object" &&
      error.response !== null &&
      "detail" in error.response &&
      typeof error.response.detail === "string"
    ) {
      return error.response.detail;
    }

    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return "Something went wrong";
}

export function useAdminCopilotPage() {
  const { toast } = useToast();
  const [search, setSearch] = useState("");
  const [selectedUser, setSelectedUser] =
    useState<AdminCopilotUserSummary | null>(null);
  const [pendingTriggerType, setPendingTriggerType] =
    useState<ChatSessionStartType | null>(null);
  const [lastTriggeredSession, setLastTriggeredSession] =
    useState<TriggerCopilotSessionResponse | null>(null);
  const [lastEmailSweepResult, setLastEmailSweepResult] =
    useState<SendCopilotEmailsResponse | null>(null);

  const deferredSearch = useDeferredValue(search);
  const normalizedSearch = deferredSearch.trim();

  const searchUsersQuery = useGetV2SearchCopilotUsers(
    normalizedSearch ? { search: normalizedSearch, limit: 20 } : undefined,
    {
      query: {
        enabled: normalizedSearch.length > 0,
        select: okData,
      },
    },
  );

  const triggerCopilotSessionMutation = usePostV2TriggerCopilotSession({
    mutation: {
      onSuccess: (response) => {
        setPendingTriggerType(null);
        const session = okData(response) ?? null;
        setLastTriggeredSession(session);
        toast({
          title: "Copilot session created",
          variant: "default",
        });
      },
      onError: (error) => {
        setPendingTriggerType(null);
        toast({
          title: getErrorMessage(error),
          variant: "destructive",
        });
      },
    },
  });

  const sendPendingCopilotEmailsMutation = useMutation({
    mutationKey: ["sendPendingCopilotEmails"],
    mutationFn: async (userId: string) =>
      customMutator<{
        data: SendCopilotEmailsResponse;
        status: number;
        headers: Headers;
      }>("/api/users/admin/copilot/send-emails", {
        method: "POST",
        body: JSON.stringify({ user_id: userId }),
      }),
    onSuccess: (response) => {
      const result = okData(response) ?? null;
      setLastEmailSweepResult(result);
      if (!result) {
        toast({
          title: "Email sweep completed",
          variant: "default",
        });
        return;
      }

      toast({
        title:
          result.sent_count > 0
            ? `Sent ${result.sent_count} Copilot email${result.sent_count === 1 ? "" : "s"}`
            : "Email sweep completed",
        description: [
          `${result.candidate_count} candidate${result.candidate_count === 1 ? "" : "s"}`,
          `${result.sent_count} sent`,
          `${result.skipped_count} skipped`,
          `${result.repair_queued_count} repairs queued`,
          `${result.running_count} still running`,
          `${result.failed_count} failed`,
        ].join(" • "),
        variant: "default",
      });
    },
    onError: (error: unknown) => {
      toast({
        title: getErrorMessage(error),
        variant: "destructive",
      });
    },
  });

  function handleSelectUser(user: AdminCopilotUserSummary) {
    setSelectedUser(user);
    setLastTriggeredSession(null);
    setLastEmailSweepResult(null);
  }

  function handleTriggerSession(startType: ChatSessionStartType) {
    if (!selectedUser) {
      return;
    }

    setPendingTriggerType(startType);
    setLastTriggeredSession(null);
    triggerCopilotSessionMutation.mutate({
      data: {
        user_id: selectedUser.id,
        start_type: startType,
      },
    });
  }

  function handleSendPendingEmails() {
    if (!selectedUser) {
      return;
    }

    setLastEmailSweepResult(null);
    sendPendingCopilotEmailsMutation.mutate(selectedUser.id);
  }

  return {
    search,
    selectedUser,
    pendingTriggerType,
    lastTriggeredSession,
    lastEmailSweepResult,
    searchedUsers: searchUsersQuery.data?.users ?? [],
    searchErrorMessage: searchUsersQuery.error
      ? getErrorMessage(searchUsersQuery.error)
      : null,
    isSearchingUsers: searchUsersQuery.isLoading,
    isRefreshingUsers:
      searchUsersQuery.isFetching && !searchUsersQuery.isLoading,
    isTriggeringSession: triggerCopilotSessionMutation.isPending,
    isSendingPendingEmails: sendPendingCopilotEmailsMutation.isPending,
    hasSearch: normalizedSearch.length > 0,
    setSearch,
    handleSelectUser,
    handleTriggerSession,
    handleSendPendingEmails,
  };
}
