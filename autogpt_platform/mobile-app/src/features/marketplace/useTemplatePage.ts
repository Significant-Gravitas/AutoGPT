import { useQuery } from "@tanstack/react-query";
import { router, useNavigation } from "expo-router";
import { useRef, useState } from "react";
import { BackHandler, type NativeEventSubscription } from "react-native";

import {
  expertKeys,
  listExperts,
  listTemplates,
  templateKeys,
} from "@/lib/api/experts";
import { isSessionGone, signOut } from "@/lib/session";
import { useMountEffect } from "@/lib/useMountEffect";

import { firstChatAfterHire, latestOrNewSession } from "./openExpertChat";
import { useHireExpert } from "./useHireExpert";

export function useTemplatePage(templateID: string) {
  const templates = useQuery({ queryKey: templateKeys.list, queryFn: listTemplates });
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });
  const [error, setError] = useState<string | null>(null);
  const [isOpeningChat, setIsOpeningChat] = useState(false);
  // State lands a render late; the ref stops a second tap in the same frame.
  const isOpeningRef = useRef(false);
  // Keyed by template, this page can be replaced while a chat is opening;
  // the old one must not push its chat over the new page.
  const isMounted = useRef(true);
  const navigation = useNavigation();
  const { hire, isHiring, justHired, newerHires } = useHireExpert({
    onHired: () => setError(null),
    onError: (message) => setError(message),
  });
  const template = templates.data?.find((item) => item.id === templateID) ?? null;
  const hiredNow = justHired(templateID);
  // As on the list: once the team list has loaded after the hire, it decides,
  // so an expert removed on the web offers Hire again.
  const hiredExpert =
    experts.data?.find((expert) => expert.source_template_id === templateID) ??
    newerHires(experts.dataUpdatedAt).get(templateID) ??
    null;

  // While this page is in front, Android's back button returns to the list
  // (the drawer's own back goes to its first screen). Leaving drops any error,
  // so a return visit starts clean.
  useMountEffect(() => {
    let backPress: NativeEventSubscription | null = null;
    function handleBackPress() {
      back();
      return true;
    }
    function handleFocus() {
      backPress?.remove();
      backPress = BackHandler.addEventListener("hardwareBackPress", handleBackPress);
    }
    function handleBlur() {
      backPress?.remove();
      backPress = null;
      setError(null);
    }
    if (navigation.isFocused()) handleFocus();
    const offFocus = navigation.addListener("focus", handleFocus);
    const offBlur = navigation.addListener("blur", handleBlur);
    return () => {
      isMounted.current = false;
      offFocus();
      offBlur();
      backPress?.remove();
    };
  });

  function handleHire() {
    if (!template) return;
    setError(null);
    void hire(template);
  }

  async function openChat() {
    if (!hiredExpert || isOpeningRef.current) return;
    isOpeningRef.current = true;
    setIsOpeningChat(true);
    setError(null);
    try {
      const sessionID = hiredNow
        ? await firstChatAfterHire(hiredExpert.id)
        : await latestOrNewSession(hiredExpert.id);
      if (isMounted.current && navigation.isFocused()) router.push(`/chat/${sessionID}`);
    } catch (caught) {
      if (isSessionGone(caught)) void signOut();
      else setError("Couldn't open that chat. Try again.");
    } finally {
      isOpeningRef.current = false;
      setIsOpeningChat(false);
    }
  }

  function back() {
    router.navigate("/marketplace");
  }

  return {
    template,
    isLoading: templates.isPending,
    // Only when nothing is cached: a failed refetch keeps the list it had.
    isLoadError: templates.isError && !templates.data,
    retry: () => void templates.refetch(),
    isOnTeam: hiredExpert !== null,
    // Hire or Chat is decided only once the team has loaded. If it can't load,
    // Hire stays offered: a hire of an expert already on the team returns them.
    isCheckingTeam: hiredExpert === null && experts.isPending,
    isHiring: isHiring(templateID),
    isOpeningChat,
    error,
    hire: handleHire,
    openChat,
    back,
  };
}
