import { useQuery } from "@tanstack/react-query";
import { router, useNavigation } from "expo-router";
import { useRef, useState } from "react";

import { useToast } from "@/components/useToast";
import {
  type Expert,
  type ExpertTemplate,
  expertKeys,
  listExperts,
  listTemplates,
  templateKeys,
} from "@/lib/api/experts";
import { isSessionGone, signOut } from "@/lib/session";

import { type Area, emptyMessage, filterTemplates, hiredTemplateIDs } from "./helpers";
import { firstChatAfterHire } from "./openExpertChat";
import { useHireExpert } from "./useHireExpert";

export function useMarketplaceScreen() {
  const templates = useQuery({ queryKey: templateKeys.list, queryFn: listTemplates });
  const experts = useQuery({ queryKey: expertKeys.list, queryFn: listExperts });
  const [query, setQuery] = useState("");
  const [area, setArea] = useState<Area | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const { toast, show, hide } = useToast();
  // Marketplace stays mounted in the drawer: someone who left meanwhile stays where they went.
  const navigation = useNavigation();
  const { hire, isHiring, justHired } = useHireExpert({
    onHired: handleHired,
    onError: handleError,
  });
  const hiredIDs = hiredTemplateIDs(experts.data ?? []);
  // State lands a render late; the ref stops a second tap in the same frame.
  const isOpeningChat = useRef(false);

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

  // Only from the toast, so always for an expert hired moments ago.
  async function openChat(expert: Expert) {
    if (isOpeningChat.current) return;
    isOpeningChat.current = true;
    show({ message: `Opening ${expert.name}'s chat…`, tone: "success" });
    try {
      const sessionID = await firstChatAfterHire(expert.id);
      hide();
      if (navigation.isFocused()) router.push(`/chat/${sessionID}`);
    } catch (caught) {
      if (isSessionGone(caught)) void signOut();
      else show({ message: "Couldn't open that chat. Try again.", tone: "error" });
    } finally {
      isOpeningChat.current = false;
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
    templates: filterTemplates(templates.data ?? [], query, area),
    isLoading: templates.isPending,
    isError: templates.isError && !templates.data,
    empty: emptyMessage(query, area),
    query,
    setQuery,
    area,
    setArea,
    isOnTeam: (template: ExpertTemplate) =>
      hiredIDs.has(template.id) || justHired(template.id) !== null,
    isHiring,
    hire,
    openTemplate,
    isRefreshing,
    refresh,
    toast,
  };
}
