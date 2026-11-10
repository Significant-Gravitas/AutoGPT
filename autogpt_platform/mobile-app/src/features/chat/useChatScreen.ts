import { skipToken, useQuery } from "@tanstack/react-query";
import { router } from "expo-router";

import {
  chatKeys,
  getSession,
  isSessionID,
  listSessions,
} from "@/lib/api/chat";
import { expertKeys, listExperts } from "@/lib/api/experts";

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

  const expertID = detail.data?.expert_id ?? null;
  const { data: experts } = useQuery({
    queryKey: expertKeys.list,
    queryFn: listExperts,
    enabled: expertID !== null,
  });

  const summary = sessions?.find((session) => session.id === sessionID);
  const expert = experts?.find((item) => item.id === expertID);
  const title = expert?.name || summary?.title || "Otto";

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
