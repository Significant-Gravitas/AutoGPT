import { useQuery } from "@tanstack/react-query";
import { router, useNavigation } from "expo-router";
import { useState } from "react";

import {
  expertKeys,
  listExperts,
  listTemplates,
  templateKeys,
} from "@/lib/api/experts";
import { isSessionGone, signOut } from "@/lib/session";

import { latestOrNewSession } from "./openExpertChat";
import { useHireExpert } from "./useHireExpert";

export function useTemplatePage(templateID: string) {
  const templates = useQuery({ queryKey: templateKeys.list, queryFn: listTemplates });
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });
  const [error, setError] = useState<string | null>(null);
  const [isOpeningChat, setIsOpeningChat] = useState(false);
  const navigation = useNavigation();
  const { hire, isHiring } = useHireExpert({
    onHired: () => setError(null),
    onError: (message) => setError(message),
  });
  const template = templates.data?.find((item) => item.id === templateID) ?? null;
  const hiredExpert =
    experts.data?.find((expert) => expert.source_template_id === templateID) ?? null;

  function handleHire() {
    if (!template) return;
    setError(null);
    void hire(template);
  }

  async function openChat() {
    if (!hiredExpert || isOpeningChat) return;
    setIsOpeningChat(true);
    setError(null);
    try {
      const sessionID = await latestOrNewSession(hiredExpert.id);
      if (navigation.isFocused()) router.push(`/chat/${sessionID}`);
    } catch (caught) {
      if (isSessionGone(caught)) void signOut();
      else setError("Couldn't open that chat. Try again.");
    } finally {
      setIsOpeningChat(false);
    }
  }

  // The drawer's own back goes to its first screen, so back is always the list.
  function back() {
    router.navigate("/marketplace");
  }

  return {
    template,
    isLoading: templates.isPending,
    isOnTeam: hiredExpert !== null,
    isHiring: isHiring(templateID),
    isOpeningChat,
    error,
    hire: handleHire,
    openChat,
    back,
  };
}
