import { skipToken, useQuery } from "@tanstack/react-query";
import { router } from "expo-router";

import {
  chatKeys,
  getSession,
  isSessionID,
  listSessions,
} from "@/lib/api/chat";

/** `routeSessionID` comes from the URL, which a deep link controls. */
export function useChatScreen(routeSessionID: unknown) {
  const sessionID = isSessionID(routeSessionID) ? routeSessionID : null;
  const detail = useQuery({
    queryKey: chatKeys.session(sessionID ?? ""),
    queryFn: sessionID ? () => getSession(sessionID) : skipToken,
    // The thread reads this once when it mounts, so a chat opened again must
    // load fresh rather than start from an old copy.
    gcTime: 0,
  });
  const { data: sessions } = useQuery({
    queryKey: chatKeys.sessions,
    queryFn: () => listSessions(),
  });

  const summary = sessions?.find((session) => session.id === sessionID);
  const title = summary?.title || "Otto";

  function newChat() {
    router.navigate("/");
  }

  return {
    sessionID,
    detail: detail.data,
    isPending: sessionID !== null && detail.isPending,
    refetch: detail.refetch,
    title,
    newChat,
  };
}
