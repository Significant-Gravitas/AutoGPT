import { useQuery } from "@tanstack/react-query";
import { router } from "expo-router";

import {
  DRAWER_SESSION_LIMIT,
  groupChatsByExpert,
} from "@/features/drawer/helpers";
import { chatKeys, listSessionsPage } from "@/lib/api/chat";
import { expertKeys, listExperts } from "@/lib/api/experts";

export function useAllChats() {
  const { data, isLoading, isError, refetch, isRefetching } = useQuery({
    queryKey: chatKeys.sessions,
    queryFn: () => listSessionsPage({ limit: DRAWER_SESSION_LIMIT }),
  });
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });

  function goBack() {
    if (router.canGoBack()) router.back();
    else router.navigate("/");
  }

  function openChat(sessionID: string) {
    router.navigate({
      pathname: "/chat/[sessionId]",
      params: { sessionId: sessionID },
    });
  }

  function refresh() {
    void refetch();
    void experts.refetch();
  }

  const sessions = data?.sessions ?? [];

  return {
    groups: groupChatsByExpert(sessions, experts.data ?? []),
    isTruncated: (data?.total ?? 0) > sessions.length,
    isLoading,
    isError,
    isRefetching,
    refresh,
    goBack,
    openChat,
  };
}
