import { useQuery } from "@tanstack/react-query";
import { router, useNavigation } from "expo-router";
import { useRef, useState } from "react";

import { useToast } from "@/components/useToast";
import { PLAN_REQUIRED_MESSAGE } from "@/features/chat/chatErrors";
import { ApiError } from "@/lib/api/client";
import {
  type Expert,
  type ExpertTemplate,
  expertKeys,
  listExperts,
  listTemplates,
  templateKeys,
} from "@/lib/api/experts";
import { isSessionGone, signOut } from "@/lib/session";

import {
  areaSections,
  filterTemplates,
  hiredTeam,
  hiredTemplateIDs,
  openSeats,
} from "./helpers";
import { firstChatAfterHire, latestOrNewSession } from "./openExpertChat";
import { useHireExpert } from "./useHireExpert";

export function useMarketplaceScreen() {
  const templates = useQuery({ queryKey: templateKeys.list, queryFn: listTemplates });
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });
  const [query, setQuery] = useState("");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [openingID, setOpeningID] = useState<string | null>(null);
  const { toast, show, hide } = useToast();
  // Marketplace stays mounted in the drawer: someone who left meanwhile stays where they went.
  const navigation = useNavigation();
  const { hire, isHiring, justHired, newerHires } = useHireExpert({
    onHired: handleHired,
    onError: handleError,
  });
  // State lands a render late; the ref stops a second tap in the same frame.
  const isOpeningChat = useRef(false);
  const all = templates.data ?? [];
  const team = hiredTeam(experts.data ?? [], newerHires(experts.dataUpdatedAt));
  const hiredIDs = hiredTemplateIDs(team);
  const sections = areaSections(all, hiredIDs);

  function handleHired(expert: Expert) {
    show({
      message: `${expert.name} joined your team`,
      tone: "success",
      actionLabel: "Chat",
      onAction: () => {
        void openChat(expert);
      },
    });
  }

  function handleError(message: string) {
    show({ message, tone: "error" });
  }

  async function openChat(expert: Expert) {
    if (isOpeningChat.current) return;
    isOpeningChat.current = true;
    setOpeningID(expert.id);
    // Hired moments ago: wait for the chat their hello lands in, which can take a while.
    const isNew =
      expert.source_template_id !== null && justHired(expert.source_template_id) !== null;
    if (isNew) show({ message: `Opening ${expert.name}'s chat…`, tone: "success" });
    try {
      const sessionID = isNew
        ? await firstChatAfterHire(expert.id)
        : await latestOrNewSession(expert.id);
      hide();
      if (navigation.isFocused()) router.push(`/chat/${sessionID}`);
    } catch (caught) {
      if (isSessionGone(caught)) {
        void signOut();
        return;
      }
      show({
        message:
          caught instanceof ApiError && caught.status === 402
            ? PLAN_REQUIRED_MESSAGE
            : "Couldn't open that chat. Try again.",
        tone: "error",
      });
    } finally {
      isOpeningChat.current = false;
      setOpeningID(null);
    }
  }

  function openTemplate(template: ExpertTemplate) {
    router.push({
      pathname: "/marketplace/[templateId]",
      params: { templateId: template.id },
    });
  }

  async function refresh() {
    setIsRefreshing(true);
    try {
      await Promise.all([templates.refetch(), experts.refetch()]);
    } finally {
      setIsRefreshing(false);
    }
  }

  return {
    hasTemplates: all.length > 0,
    isLoading: templates.isPending,
    isError: templates.isError && !templates.data,
    query,
    setQuery,
    isSearching: query.trim() !== "",
    matches: filterTemplates(all, query),
    sections,
    team,
    openSeats: openSeats(sections),
    openingID,
    openChat,
    isOnTeam: (template: ExpertTemplate) => hiredIDs.has(template.id),
    isHiring,
    hire,
    openTemplate,
    isRefreshing,
    refresh,
    toast,
  };
}
