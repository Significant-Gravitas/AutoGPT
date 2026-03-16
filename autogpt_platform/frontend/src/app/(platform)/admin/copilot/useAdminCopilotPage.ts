"use client";

import type { AdminCopilotUserSummary } from "@/app/api/__generated__/models/adminCopilotUserSummary";
import { ChatSessionStartType } from "@/app/api/__generated__/models/chatSessionStartType";
import type { TriggerCopilotSessionResponse } from "@/app/api/__generated__/models/triggerCopilotSessionResponse";
import { okData } from "@/app/api/helpers";
import {
  useGetV2SearchCopilotUsers,
  usePostV2TriggerCopilotSession,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
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

  function handleSelectUser(user: AdminCopilotUserSummary) {
    setSelectedUser(user);
    setLastTriggeredSession(null);
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

  return {
    search,
    selectedUser,
    pendingTriggerType,
    lastTriggeredSession,
    searchedUsers: searchUsersQuery.data?.users ?? [],
    searchErrorMessage: searchUsersQuery.error
      ? getErrorMessage(searchUsersQuery.error)
      : null,
    isSearchingUsers: searchUsersQuery.isLoading,
    isRefreshingUsers:
      searchUsersQuery.isFetching && !searchUsersQuery.isLoading,
    isTriggeringSession: triggerCopilotSessionMutation.isPending,
    hasSearch: normalizedSearch.length > 0,
    setSearch,
    handleSelectUser,
    handleTriggerSession,
  };
}
