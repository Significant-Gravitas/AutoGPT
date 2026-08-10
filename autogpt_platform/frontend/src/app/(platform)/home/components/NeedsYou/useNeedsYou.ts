import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useProcessReviews } from "@/hooks/useProcessReviews";

interface Args {
  items: HomeAttentionItem[];
}

type AttentionFilter = "all" | HomeAttentionItem["kind"];

const FILTER_LABELS: Partial<Record<AttentionFilter, string>> = {
  all: "All",
  approval: "Approvals",
  setup: "Setup",
  paused: "Paused",
  credits: "Credits",
};

export function useNeedsYou({ items }: Args) {
  const queryClient = useQueryClient();
  const { processReviews } = useProcessReviews();
  const { toast } = useToast();
  const [showAll, setShowAll] = useState(false);
  const [activeKind, setActiveKind] = useState<
    "all" | HomeAttentionItem["kind"]
  >("all");
  const [pendingIDs, setPendingIDs] = useState<Set<string>>(new Set());
  const filterKinds = Array.from(new Set(items.map((item) => item.kind)));
  const selectedKind: "all" | HomeAttentionItem["kind"] =
    activeKind !== "all" && filterKinds.includes(activeKind)
      ? activeKind
      : "all";
  const filteredItems =
    selectedKind === "all"
      ? items
      : items.filter((item) => item.kind === selectedKind);
  const visibleItems = showAll ? filteredItems : filteredItems.slice(0, 3);

  function selectKind(kind: "all" | HomeAttentionItem["kind"]) {
    setActiveKind(kind);
    setShowAll(false);
  }

  function setPending(itemID: string, pending: boolean) {
    setPendingIDs((current) => {
      const next = new Set(current);
      if (pending) next.add(itemID);
      else next.delete(itemID);
      return next;
    });
  }

  async function decide(item: HomeAttentionItem, approved: boolean) {
    if (!item.review) return;
    setPending(item.id, true);
    try {
      const response = await processReviews(
        [
          {
            node_exec_id: item.review.node_exec_id,
            approved,
            auto_approve_future: false,
          },
        ],
        [item.review.graph_exec_id],
      );
      if (response.status !== 200 || response.data.failed_count > 0) {
        const message = response.status === 200 ? response.data.error : null;
        throw new Error(message || "The review could not be processed.");
      }
      toast({ title: approved ? "Approved" : "Declined" });
      await queryClient.invalidateQueries({
        queryKey: getGetHomeDashboardQueryKey(),
      });
    } catch (error) {
      toast({
        title: approved ? "Could not approve" : "Could not decline",
        description: error instanceof Error ? error.message : "Try again.",
        variant: "destructive",
      });
    } finally {
      setPending(item.id, false);
    }
  }

  return {
    filteredItems,
    visibleItems,
    filterOptions: (["all", ...filterKinds] as AttentionFilter[]).map(
      (value) => ({ value, label: FILTER_LABELS[value] ?? value }),
    ),
    hasFilters: filterKinds.length > 1,
    selectedKind,
    selectKind,
    showAll,
    setShowAll,
    pendingIDs,
    decide,
  };
}
