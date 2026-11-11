import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { PLAN_REQUIRED_MESSAGE } from "@/features/chat/chatErrors";
import { chatKeys, getSuggestedPrompts } from "@/lib/api/chat";
import { ApiError } from "@/lib/api/client";
import { expertKeys, listExperts } from "@/lib/api/experts";
import { getHome, homeKeys } from "@/lib/api/home";
import { authClient } from "@/lib/auth/client";
import { OTTO_NAME } from "@/lib/otto";
import { isSessionGone, signOut } from "@/lib/session";

import { briefingLine, greeting } from "./helpers";
import { startChat } from "./startChat";

const FALLBACK_PROMPTS = [
  "Plan my week",
  "Draft a follow-up email",
  "Research a topic for me",
  "Sum up a long document",
];

export function useHomeScreen() {
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  // null means Otto.
  const [selectedID, setSelectedID] = useState<string | null>(null);
  // State lands a render late; the ref stops a second tap in the same frame.
  const isStartingRef = useRef(false);
  const { data } = useQuery({
    queryKey: chatKeys.suggestedPrompts,
    queryFn: getSuggestedPrompts,
  });
  const home = useQuery({ queryKey: homeKeys.dashboard, queryFn: getHome });
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });
  const { data: session } = authClient.useSession();

  const suggested = (data ?? []).filter(Boolean).slice(0, 6);
  const prompts = suggested.length > 0 ? suggested : FALLBACK_PROMPTS;
  const team = experts.data ?? [];
  const selected = team.find((expert) => expert.id === selectedID) ?? null;
  const dashboard = home.isError ? undefined : home.data;

  async function refresh() {
    setIsRefreshing(true);
    try {
      await Promise.all([home.refetch(), experts.refetch()]);
    } finally {
      setIsRefreshing(false);
    }
  }

  async function submit(text: string) {
    if (isStartingRef.current) return false;
    isStartingRef.current = true;
    setIsStarting(true);
    setError(null);
    try {
      await startChat(text, selected?.id);
      return true;
    } catch (caught) {
      if (isSessionGone(caught)) {
        void signOut();
        return false;
      }
      setError(
        caught instanceof ApiError && caught.status === 402
          ? PLAN_REQUIRED_MESSAGE
          : "Couldn't start a chat. Try again.",
      );
      return false;
    } finally {
      isStartingRef.current = false;
      setIsStarting(false);
    }
  }

  return {
    heading: greeting(session?.user?.name),
    subline: briefingLine(dashboard?.briefing.completed_count),
    attention: dashboard?.attention ?? [],
    activeTask: dashboard?.active_tasks[0] ?? null,
    upcomingTask: dashboard?.upcoming_tasks[0] ?? null,
    experts: team,
    selectedID: selected?.id ?? null,
    select: setSelectedID,
    placeholder: selected ? `Message ${selected.name}` : `Ask ${OTTO_NAME} anything`,
    prompts,
    isStarting,
    error,
    isRefreshing,
    refresh,
    submit,
  };
}
