import { useQuery } from "@tanstack/react-query";
import { router, usePathname } from "expo-router";
import type { DrawerContentComponentProps } from "expo-router/drawer";

import { chatKeys, listSessions } from "@/lib/api/chat";
import { authClient } from "@/lib/auth/client";
import { signOut } from "@/lib/session";

const CHAT_PATH_PREFIX = "/chat/";

export function useAppDrawer(navigation: DrawerContentComponentProps["navigation"]) {
  const pathname = usePathname();
  const { data: session } = authClient.useSession();
  const { data, isLoading, isError, refetch, isRefetching } = useQuery({
    queryKey: chatKeys.sessions,
    queryFn: () => listSessions(),
  });

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

  return {
    sessions: data ?? [],
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
