import { useQuery } from "@tanstack/react-query";
import { router, usePathname } from "expo-router";
import type { DrawerContentComponentProps } from "expo-router/drawer";
import { useState } from "react";

import {
  DRAWER_SESSION_LIMIT,
  groupChatsByExpert,
  searchChats,
} from "@/features/drawer/helpers";
import { chatKeys, listSessionsPage } from "@/lib/api/chat";
import { expertKeys, listExperts } from "@/lib/api/experts";
import { useSession } from "@/lib/auth/nativeSession";
import { signOut } from "@/lib/session";
import { openHome } from "@/features/home/openHome";

const CHAT_PATH_PREFIX = "/chat/";

export function useAppDrawer(
  navigation: DrawerContentComponentProps["navigation"],
) {
  const pathname = usePathname();
  const session = useSession();
  const [query, setQuery] = useState("");
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const { data, isLoading, isError, refetch, isRefetching } = useQuery({
    queryKey: chatKeys.sessions,
    queryFn: () => listSessionsPage({ limit: DRAWER_SESSION_LIMIT }),
  });
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });

  const activeSessionID = pathname.startsWith(CHAT_PATH_PREFIX)
    ? pathname.slice(CHAT_PATH_PREFIX.length)
    : undefined;

  function openSearch() {
    setIsSearchOpen(true);
  }

  function closeSearch() {
    setQuery("");
    setIsSearchOpen(false);
  }

  function openChat(sessionID: string) {
    // The next time the menu opens it shows its rows, not these results.
    closeSearch();
    router.navigate({
      pathname: "/chat/[sessionId]",
      params: { sessionId: sessionID },
    });
    navigation.closeDrawer();
  }

  function newChat() {
    openHome(null);
    navigation.closeDrawer();
  }

  function openTeam() {
    router.navigate("/team");
    navigation.closeDrawer();
  }

  function openMarketplace() {
    router.navigate("/marketplace");
    navigation.closeDrawer();
  }

  function openAllChats() {
    router.navigate("/chats");
    navigation.closeDrawer();
  }

  function openSettings() {
    router.navigate("/settings");
    navigation.closeDrawer();
  }

  // No closeDrawer: signing out takes this drawer off the screen at once.
  function handleSignOut() {
    void signOut();
  }

  const groups = groupChatsByExpert(data?.sessions ?? [], experts.data ?? []);

  // Names and avatars come from the experts list, so a pull refreshes both.
  function refreshAll() {
    return Promise.all([refetch(), experts.refetch()]);
  }

  return {
    query,
    setQuery,
    isSearchOpen,
    openSearch,
    closeSearch,
    isSearching: query.trim().length > 0,
    groups,
    matches: searchChats(groups, query),
    isLoading,
    isError,
    refetch: refreshAll,
    isRefetching,
    email: session?.user.email,
    userName: session?.user.name,
    userImage: session?.user.image,
    activeSessionID,
    openChat,
    newChat,
    openTeam,
    openMarketplace,
    openAllChats,
    openSettings,
    handleSignOut,
  };
}
