import { useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { PLAN_REQUIRED_MESSAGE } from "@/features/chat/chatErrors";
import { useAttachments } from "@/features/chat/useAttachments";
import { chatKeys, getSuggestedPrompts } from "@/lib/api/chat";
import { ApiError } from "@/lib/api/client";
import { expertKeys, listExperts } from "@/lib/api/experts";
import { expertTint } from "@/lib/expertPalette";
import { getHome, homeKeys } from "@/lib/api/home";
import { authClient } from "@/lib/auth/client";
import { isSessionGone, signOut } from "@/lib/session";

import { areaPrompts } from "./areaPrompts";
import {
  firstName,
  hasExistingWork,
  inputPlaceholder,
  ottoWording,
} from "./helpers";
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
  // No chat exists until the first send, so files upload to the workspace.
  const files = useAttachments(null);

  const suggested = (data ?? []).filter(Boolean).slice(0, 6).map(ottoWording);
  const team = experts.data ?? [];
  const selected = team.find((expert) => expert.id === selectedID) ?? null;
  // The suggestions come from what Otto knows about the user's business; an
  // expert gets starters for their own area instead.
  const ottoPrompts = suggested.length > 0 ? suggested : FALLBACK_PROMPTS;
  const prompts = selected ? areaPrompts(selected) : ottoPrompts;
  const promptExpert = selected
    ? {
        name: selected.name,
        tint: expertTint({
          avatarUrl: selected.avatar_url,
          categories: selected.categories,
          role: selected.role,
        }),
      }
    : null;
  // A failed refresh keeps the last good answer on screen.
  const dashboard = home.data;
  const attention = dashboard?.attention ?? [];
  const activeTask = dashboard?.active_tasks[0] ?? null;
  const upcomingTask = dashboard?.upcoming_tasks[0] ?? null;

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
      await startChat(text, selected?.id, files.readyFiles);
      files.clear();
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
    name: firstName(session?.user?.name, session?.user?.email),
    attention,
    activeTask,
    upcomingTask,
    hasCards: attention.length > 0 || !!activeTask || !!upcomingTask,
    experts: team,
    selected,
    select: setSelectedID,
    placeholder: inputPlaceholder(selected, hasExistingWork(dashboard)),
    prompts,
    promptExpert,
    files,
    isStarting,
    error,
    isRefreshing,
    refresh,
    submit,
  };
}
