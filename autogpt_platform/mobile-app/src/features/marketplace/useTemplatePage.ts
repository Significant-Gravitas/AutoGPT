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

import { latestOrNewSession } from "./openExpertChat";
import { useHireExpert } from "./useHireExpert";

export function useTemplatePage(templateID: string) {
  const templates = useQuery({ queryKey: templateKeys.list, queryFn: listTemplates });
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });
  const [error, setError] = useState<string | null>(null);
  const [isOpeningChat, setIsOpeningChat] = useState(false);
  // State lands a render late; the ref stops a second tap in the same frame.
  const isOpeningRef = useRef(false);
  const navigation = useNavigation();
  const { hire, isHiring } = useHireExpert({
    onHired: () => setError(null),
    onError: (message) => setError(message),
  });
  const template = templates.data?.find((item) => item.id === templateID) ?? null;
  const hiredExpert =
    experts.data?.find((expert) => expert.source_template_id === templateID) ?? null;

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
      const sessionID = await latestOrNewSession(hiredExpert.id);
      if (navigation.isFocused()) router.push(`/chat/${sessionID}`);
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
    isOnTeam: hiredExpert !== null,
    isHiring: isHiring(templateID),
    isOpeningChat,
    error,
    hire: handleHire,
    openChat,
    back,
  };
}
