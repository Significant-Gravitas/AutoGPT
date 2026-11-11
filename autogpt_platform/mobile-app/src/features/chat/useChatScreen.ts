import { skipToken, useQuery } from "@tanstack/react-query";
import { router } from "expo-router";

import {
  chatKeys,
  getSession,
  isSessionID,
} from "@/lib/api/chat";
import { expertKeys, listExperts } from "@/lib/api/experts";
import { avatarSource, roleLine } from "@/features/team/helpers";
import { OTTO_AVATAR_URL, OTTO_NAME, OTTO_ROLE } from "@/lib/otto";

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
  const expertID = detail.data?.expert_id ?? null;
  const experts = useQuery({
    queryKey: expertKeys.list,
    queryFn: listExperts,
    enabled: expertID !== null,
  });

  function readIdentity() {
    if (!detail.data) return null;
    if (!expertID) {
      return { name: OTTO_NAME, role: OTTO_ROLE, avatar: OTTO_AVATAR_URL };
    }
    if (experts.isPending) return null;
    const expert = experts.data?.find((item) => item.id === expertID);
    if (!expert) return { name: "Expert", role: null, avatar: null };
    return {
      name: expert.name,
      role: roleLine(expert),
      avatar: avatarSource(expert),
    };
  }

  function newChat() {
    router.navigate("/");
  }

  return {
    sessionID,
    detail: detail.data,
    isPending: sessionID !== null && detail.isPending,
    refetch: detail.refetch,
    identity: readIdentity(),
    newChat,
  };
}
