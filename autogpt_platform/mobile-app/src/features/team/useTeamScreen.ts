import { useQuery } from "@tanstack/react-query";
import { router, useNavigation } from "expo-router";
import { useRef, useState } from "react";

import { PLAN_REQUIRED_MESSAGE } from "@/features/chat/chatErrors";
import { chooseAction } from "@/lib/actionMenu";
import { tapMenu } from "@/lib/haptics";
import { latestOrNewSession } from "@/features/marketplace/openExpertChat";
import { ApiError } from "@/lib/api/client";
import { type Expert, expertKeys, listExperts } from "@/lib/api/experts";
import { isSessionGone, signOut } from "@/lib/session";
import { openHome } from "@/features/home/openHome";

export function useTeamScreen() {
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [openingID, setOpeningID] = useState<string | null>(null);
  const [openError, setOpenError] = useState<string | null>(null);
  // State lands a render late; the ref stops a second tap in the same frame.
  const isOpening = useRef(false);
  const navigation = useNavigation();

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
      // Team stays mounted in the drawer: someone who left meanwhile stays where they went.
      if (navigation.isFocused()) router.push(`/chat/${sessionID}`);
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

  async function openExpertMenu(expert: Expert) {
    tapMenu();
    const action = await chooseAction(expert.name, [
      { id: "chat", label: "Chat" },
      { id: "manage", label: "Manage" },
    ]);
    if (action === "chat") void openExpert(expert);
    if (action === "manage") router.push(`/expert/${expert.id}`);
  }

  function openOtto() {
    openHome(null);
  }

  function createExpert() {
    router.navigate("/create-expert");
  }

  function browseMarketplace() {
    router.navigate("/marketplace");
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
    openExpertMenu,
    openOtto,
    createExpert,
    browseMarketplace,
  };
}
