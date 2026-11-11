import { useQuery } from "@tanstack/react-query";
import { router, usePathname } from "expo-router";
import type { DrawerContentComponentProps } from "expo-router/drawer";

import {
  DRAWER_SESSION_LIMIT,
  groupChatsByExpert,
} from "@/features/drawer/helpers";
import { chatKeys, listSessionsPage } from "@/lib/api/chat";
import { expertKeys, listExperts } from "@/lib/api/experts";
import { authClient } from "@/lib/auth/client";
import { signOut } from "@/lib/session";

const CHAT_PATH_PREFIX = "/chat/";

export function useAppDrawer(navigation: DrawerContentComponentProps["navigation"]) {
  const pathname = usePathname();
  const { data: session } = authClient.useSession();
  const { data, isLoading, isError, refetch, isRefetching } = useQuery({
    queryKey: chatKeys.sessions,
    queryFn: () => listSessionsPage({ limit: DRAWER_SESSION_LIMIT }),
  });
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });

  const activeSessionID = pathname.startsWith(CHAT_PATH_PREFIX)
    ? pathname.slice(CHAT_PATH_PREFIX.length)
    : undefined;

  function openChat(sessionID: string) {
    router.navigate({
      pathname: "/chat/[sessionId]",
      params: { sessionId: sessionID },
    });
    navigation.closeDrawer();
  }

  function newChat() {
    router.navigate("/");
    navigation.closeDrawer();
  }

  function openTeam() {
    router.navigate("/team");
    navigation.closeDrawer();
  }

  function openSettings() {
    router.navigate("/settings");
    navigation.closeDrawer();
  }

  function handleSignOut() {
    navigation.closeDrawer();
    void signOut();
  }

  const sessions = data?.sessions ?? [];

  return {
    groups: groupChatsByExpert(sessions, experts.data ?? []),
    isTruncated: (data?.total ?? 0) > sessions.length,
    isLoading,
    isError,
    refetch,
    isRefetching,
    email: session?.user.email,
    activeSessionID,
    openChat,
    newChat,
    openTeam,
    openSettings,
    handleSignOut,
  };
}
