import { useQuery } from "@tanstack/react-query";
import { router } from "expo-router";
import { useRef, useState } from "react";

import { PLAN_REQUIRED_MESSAGE } from "@/features/chat/chatErrors";
import { chatKeys, createSession, listSessions } from "@/lib/api/chat";
import { ApiError } from "@/lib/api/client";
import { type Expert, expertKeys, listExperts } from "@/lib/api/experts";
import { isSessionGone, queryClient, signOut } from "@/lib/session";

async function latestOrNewSession(expertID: string) {
  const [latest] = await listSessions({
    expertID,
    limit: 1,
    pinnedFirst: false,
  });
  if (latest) return latest.id;
  const { id } = await createSession(expertID);
  void queryClient.invalidateQueries({ queryKey: chatKeys.sessions });
  return id;
}

export function useTeamScreen() {
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [openingID, setOpeningID] = useState<string | null>(null);
  const [openError, setOpenError] = useState<string | null>(null);
  // State lands a render late; the ref stops a second tap in the same frame.
  const isOpening = useRef(false);

  async function refresh() {
    setIsRefreshing(true);
    try {
      await experts.refetch();
    } finally {
      setIsRefreshing(false);
    }
  }

  async function openExpert(expert: Expert) {
    if (isOpening.current) return;
    isOpening.current = true;
    setOpeningID(expert.id);
    setOpenError(null);
    try {
      const sessionID = await latestOrNewSession(expert.id);
      router.push(`/chat/${sessionID}`);
    } catch (caught) {
      if (isSessionGone(caught)) {
        void signOut();
        return;
      }
      setOpenError(
        caught instanceof ApiError && caught.status === 402
          ? PLAN_REQUIRED_MESSAGE
          : "Couldn't open that chat. Try again.",
      );
    } finally {
      isOpening.current = false;
      setOpeningID(null);
    }
  }

  function openOtto() {
    router.navigate("/");
  }

  function createExpert() {
    router.navigate("/create-expert");
  }

  return {
    experts: experts.data ?? [],
    isLoading: experts.isPending,
    isError: experts.isError && !experts.data,
    isRefreshing,
    refresh,
    openingID,
    openError,
    openExpert,
    openOtto,
    createExpert,
  };
}
