import { useQuery } from "@tanstack/react-query";
import { useLocalSearchParams } from "expo-router";
import { useRef, useState } from "react";

import { isUploading } from "@/features/chat/attachmentHelpers";
import { PLAN_REQUIRED_MESSAGE } from "@/features/chat/chatErrors";
import { useAttachments } from "@/features/chat/useAttachments";
import { chatKeys, getSuggestedPrompts } from "@/lib/api/chat";
import { ApiError } from "@/lib/api/client";
import { expertKeys, listExperts } from "@/lib/api/experts";
import { getHome, homeKeys } from "@/lib/api/home";
import { isSessionGone, signOut } from "@/lib/session";

import { areaPrompts } from "./areaPrompts";
import { recipientFromParam } from "./openHome";
import {
  hasExistingWork,
  homeRows,
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
  const params = useLocalSearchParams<{ recipient?: string; at?: string }>();
  // null means Otto.
  const [selectedID, setSelectedID] = useState<string | null>(() =>
    recipientFromParam(params.recipient),
  );
  // Home stays mounted, so a later "new chat with this expert" link arrives as
  // new params: pick its recipient once per request.
  const [handledAt, setHandledAt] = useState(params.at);
  if (params.at !== handledAt) {
    setHandledAt(params.at);
    setSelectedID(recipientFromParam(params.recipient));
  }
  // State lands a render late; the ref stops a second tap in the same frame.
  const isStartingRef = useRef(false);
  const { data } = useQuery({
    queryKey: chatKeys.suggestedPrompts,
    queryFn: getSuggestedPrompts,
  });
  const home = useQuery({ queryKey: homeKeys.dashboard, queryFn: getHome });
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });
  // No chat exists until the first send, so files upload to the workspace.
  const files = useAttachments(null);

  const suggested = (data ?? []).filter(Boolean).slice(0, 6).map(ottoWording);
  const team = experts.data ?? [];
  const selected = team.find((expert) => expert.id === selectedID) ?? null;
  // The suggestions come from what Otto knows about the user's business; an
  // expert gets starters for their own area instead.
  const ottoPrompts = suggested.length > 0 ? suggested : FALLBACK_PROMPTS;
  // A failed refresh keeps the last good answer on screen.
  const dashboard = home.data;
  // What needs you belongs to the whole team, so it shows only with Otto.
  const rows = selected
    ? homeRows({ prompts: areaPrompts(selected) })
    : homeRows({
        attention: dashboard?.attention,
        activeTask: dashboard?.active_tasks[0],
        upcomingTask: dashboard?.upcoming_tasks[0],
        prompts: ottoPrompts,
      });

  async function refresh() {
    setIsRefreshing(true);
    try {
      await Promise.all([home.refetch(), experts.refetch()]);
    } finally {
      setIsRefreshing(false);
    }
  }

  /**
   * Who a send goes to: null for Otto, undefined when it can't go yet. A
   * "new chat with this expert" link can name someone the team list doesn't
   * have yet (still loading, or cached from before they were hired), so look
   * again rather than send it to Otto.
   */
  async function recipientID() {
    if (selectedID === null || selected) return selected?.id ?? null;
    const asked = selectedID;
    const fresh = await experts.refetch();
    const found = fresh.data?.find((expert) => expert.id === asked);
    if (found) return found.id;
    if (fresh.isError) {
      setError("Couldn't load your team. Try again.");
    } else {
      // Gone from the team. Someone picked meanwhile keeps their place.
      setSelectedID((current) => (current === asked ? null : current));
      setError("That expert isn't on your team any more, so nothing was sent.");
    }
    return undefined;
  }

  async function submit(text: string) {
    if (isStartingRef.current) return false;
    // A starter row skips the composer's own check, and the send would leave the file behind.
    if (isUploading(files.attachments)) {
      setError("Wait for your files to finish uploading.");
      return false;
    }
    isStartingRef.current = true;
    setIsStarting(true);
    setError(null);
    try {
      const expertID = await recipientID();
      if (expertID === undefined) return false;
      await startChat(text, expertID ?? undefined, files.readyFiles);
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
    experts: team,
    selected,
    select: setSelectedID,
    placeholder: inputPlaceholder(selected, hasExistingWork(dashboard)),
    rows,
    files,
    isStarting,
    error,
    isRefreshing,
    refresh,
    submit,
  };
}
