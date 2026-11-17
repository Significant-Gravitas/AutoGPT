import { useQuery } from "@tanstack/react-query";
import { router, useNavigation } from "expo-router";
import { useState } from "react";

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
import { latestOrNewSession } from "./openExpertChat";
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
  const { hire, isHiring } = useHireExpert({ onHired: handleHired, onError: handleError });
  const hiredIDs = hiredTemplateIDs(experts.data ?? []);

  function handleHired(expert: Expert) {
    show({
      message: `${expert.name} joined your team`,
      tone: "success",
      actionLabel: "Chat",
      onAction: () => {
        hide();
        void openChat(expert.id);
      },
    });
  }

  function handleError(message: string) {
    show({ message, tone: "error" });
  }

  async function openChat(expertID: string) {
    try {
      const sessionID = await latestOrNewSession(expertID);
      if (navigation.isFocused()) router.push(`/chat/${sessionID}`);
    } catch (caught) {
      if (isSessionGone(caught)) void signOut();
      else show({ message: "Couldn't open that chat. Try again.", tone: "error" });
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
    isOnTeam: (template: ExpertTemplate) => hiredIDs.has(template.id),
    isHiring,
    hire,
    openTemplate,
    isRefreshing,
    refresh,
    toast,
  };
}
