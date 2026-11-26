import { useMutation, useQueryClient } from "@tanstack/react-query";
import { router } from "expo-router";
import { useRef, useState } from "react";
import { Linking } from "react-native";

import type { ToastState } from "@/components/useToast";
import { chatKeys } from "@/lib/api/chat";
import { ApiError } from "@/lib/api/client";
import { expertActivityKeys } from "@/lib/api/expertActivity";
import { homeKeys } from "@/lib/api/home";
import { resumeScheduledChat } from "@/lib/api/scheduledChats";
import { FRONTEND_URL } from "@/lib/env";

import { lowerFirst, noticeLinkTarget, type ChatNotice } from "../../notice";

interface Args {
  notice: ChatNotice;
  sessionID: string;
  onToast: (toast: ToastState) => void;
}

export function useNoticeCard({ notice, sessionID, onToast }: Args) {
  const queryClient = useQueryClient();
  const [connectProvider, setConnectProvider] = useState<string | null>(null);
  const [isLessOftenOpen, setIsLessOftenOpen] = useState(false);
  const [isResumed, setIsResumed] = useState(false);
  // After a reconnect there may be nothing to resume: a chat that was only
  // warned, never paused, answers 404, and that needs no word.
  const isQuiet = useRef(false);

  function refresh() {
    void queryClient.invalidateQueries({ queryKey: expertActivityKeys.graphSchedules });
    void queryClient.invalidateQueries({ queryKey: expertActivityKeys.followupSchedules });
    // Home lists each paused chat under what needs you.
    void queryClient.invalidateQueries({ queryKey: homeKeys.dashboard });
    void queryClient.invalidateQueries({ queryKey: chatKeys.session(sessionID) });
  }

  const resumeMutation = useMutation({
    mutationFn: resumeScheduledChat,
    onSuccess() {
      setIsResumed(true);
      refresh();
      onToast({ message: "Resumed. It runs again at its next scheduled time.", tone: "success" });
    },
    onError(error) {
      if (error instanceof ApiError && error.status === 404) {
        setIsResumed(true);
        refresh();
        if (!isQuiet.current) onToast({ message: "Already running", tone: "success" });
        return;
      }
      onToast({ message: "Couldn't resume this scheduled chat. Try again.", tone: "error" });
    },
  });

  function resume(quiet: boolean) {
    if (!notice.scheduleKey || resumeMutation.isPending || isResumed) return;
    isQuiet.current = quiet;
    resumeMutation.mutate(notice.scheduleKey);
  }

  function handleResume() {
    resume(false);
  }

  function openLink(href: string) {
    const target = noticeLinkTarget(href, FRONTEND_URL);
    if (target.kind === "plan") {
      router.navigate("/plan");
      return;
    }
    Linking.openURL(target.url).catch(() => undefined);
  }

  function closeConnect() {
    setConnectProvider(null);
  }

  // Reconnecting is the fix, so it resumes the chat too.
  function handleConnected() {
    setConnectProvider(null);
    resume(true);
  }

  function openLessOften() {
    setIsLessOftenOpen(true);
  }

  function closeLessOften() {
    setIsLessOftenOpen(false);
  }

  function handleCadenceSaved(label: string) {
    setIsLessOftenOpen(false);
    onToast({ message: `Now runs ${lowerFirst(label)}`, tone: "success" });
  }

  return {
    isResumed,
    isResuming: resumeMutation.isPending,
    handleResume,
    openLink,
    connectProvider,
    openConnect: setConnectProvider,
    closeConnect,
    handleConnected,
    isLessOftenOpen,
    openLessOften,
    closeLessOften,
    handleCadenceSaved,
  };
}
