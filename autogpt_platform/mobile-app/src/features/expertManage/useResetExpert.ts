import { type QueryClient, useQueryClient } from "@tanstack/react-query";
import { router } from "expo-router";
import { useState } from "react";

import { latestOrNewSession } from "@/features/marketplace/openExpertChat";
import { chatKeys, type SessionDetail } from "@/lib/api/chat";
import { expertActivityKeys } from "@/lib/api/expertActivity";
import { resetExpert } from "@/lib/api/expertManage";
import { expertKeys } from "@/lib/api/experts";
import { homeKeys } from "@/lib/api/home";

import { resetOutcome } from "./dangerHelpers";

type Phase = "confirm" | "busy" | "done";

interface Args {
  expertID: string;
  name: string;
  isOpen: boolean;
  onReset: () => void;
  onClose: () => void;
}

export function useResetExpert({ expertID, name, isOpen, onReset, onClose }: Args) {
  const queryClient = useQueryClient();
  const [phase, setPhase] = useState<Phase>("confirm");
  const [isResetting, setIsResetting] = useState(false);
  const [isOpeningChat, setIsOpeningChat] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [wasOpen, setWasOpen] = useState(isOpen);

  // Each opening starts at the confirm step, whatever the last one ended on.
  if (isOpen !== wasOpen) {
    setWasOpen(isOpen);
    if (isOpen) {
      setPhase("confirm");
      setError(null);
    }
  }

  async function reset() {
    if (isResetting) return;
    setIsResetting(true);
    setError(null);
    try {
      await resetExpert(expertID);
    } catch (caught) {
      setIsResetting(false);
      // The sheet stays open, so the error shows in it: a toast would sit behind.
      if (resetOutcome(caught) === "busy") setPhase("busy");
      else setError(`Could not finish resetting ${name}`);
      return;
    }
    setIsResetting(false);
    setPhase("done");
    onReset();
    forgetExpert(queryClient, expertID);
  }

  // The reset removed every chat, so the next one is new.
  async function openChat() {
    if (isOpeningChat) return;
    setIsOpeningChat(true);
    setError(null);
    try {
      const sessionID = await latestOrNewSession(expertID);
      onClose();
      router.push(`/chat/${sessionID}`);
    } catch {
      setError("Couldn't open that chat. Try again.");
    } finally {
      setIsOpeningChat(false);
    }
  }

  function confirm() {
    void reset();
  }

  function chat() {
    void openChat();
  }

  return { phase, isResetting, isOpeningChat, error, confirm, chat };
}

function forgetExpert(queryClient: QueryClient, expertID: string) {
  queryClient.removeQueries({
    queryKey: ["chat", "session"],
    predicate: (query) =>
      (query.state.data as SessionDetail | undefined)?.expert_id === expertID,
  });
  for (const queryKey of [
    expertActivityKeys.detail(expertID),
    expertActivityKeys.runs(expertID),
    expertActivityKeys.computer(expertID),
    expertActivityKeys.graphSchedules,
    expertActivityKeys.followupSchedules,
    chatKeys.sessions,
    homeKeys.dashboard,
  ]) {
    void queryClient.invalidateQueries({ queryKey });
  }
  void queryClient.invalidateQueries({ queryKey: expertKeys.list, exact: true });
}
