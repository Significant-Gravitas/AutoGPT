import { type QueryClient, useQueryClient } from "@tanstack/react-query";
import { router } from "expo-router";
import { useState } from "react";

import { latestOrNewSession } from "@/features/marketplace/openExpertChat";
import { chatKeys, type SessionDetail } from "@/lib/api/chat";
import { expertActivityKeys } from "@/lib/api/expertActivity";
import { resetExpert } from "@/lib/api/expertManage";
import { expertKeys } from "@/lib/api/experts";
import { homeKeys } from "@/lib/api/home";
import { sessionGeneration } from "@/lib/session";

import { isGoneError, resetOutcome } from "./dangerHelpers";

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

  if (isOpen !== wasOpen) {
    setWasOpen(isOpen);
    if (isOpen) {
      setPhase("confirm");
      setError(null);
    }
  }

  async function reset() {
    if (isResetting) return;
    const generation = sessionGeneration();
    setIsResetting(true);
    setError(null);
    try {
      await resetExpert(expertID);
    } catch (caught) {
      if (generation !== sessionGeneration()) return;
      setIsResetting(false);
      if (isGoneError(caught)) {
        leave();
        return;
      }
      if (resetOutcome(caught) === "busy") {
        setPhase("busy");
        return;
      }
      setPhase("confirm");
      setError(`Could not finish resetting ${name}`);
      return;
    }
    if (generation !== sessionGeneration()) return;
    setIsResetting(false);
    setPhase("done");
    onReset();
    forgetExpert(queryClient, expertID);
  }

  function leave() {
    onClose();
    void queryClient.invalidateQueries({ queryKey: expertKeys.list });
    router.navigate("/team");
  }

  async function openChat() {
    if (isOpeningChat) return;
    const generation = sessionGeneration();
    setIsOpeningChat(true);
    setError(null);
    try {
      const sessionID = await latestOrNewSession(expertID);
      if (generation !== sessionGeneration()) return;
      onClose();
      router.push(`/chat/${sessionID}`);
    } catch {
      if (generation === sessionGeneration()) setError("Couldn't open that chat. Try again.");
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
