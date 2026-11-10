import { useQuery } from "@tanstack/react-query";
import { router } from "expo-router";

import { chatKeys, getSession, listSessions } from "@/lib/api/chat";

export function useChatScreen(sessionID: string) {
  const detail = useQuery({
    queryKey: chatKeys.session(sessionID),
    queryFn: () => getSession(sessionID),
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
    detail: detail.data,
    isPending: detail.isPending,
    refetch: detail.refetch,
    title,
    newChat,
  };
}
