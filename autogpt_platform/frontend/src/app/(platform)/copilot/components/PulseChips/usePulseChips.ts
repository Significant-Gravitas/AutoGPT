"use client";

import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";
import { useSitrepItems } from "@/app/(platform)/library/components/SitrepItem/useSitrepItems";
import type { PulseChipData } from "./types";
import { useMemo } from "react";

const THREE_DAYS_MS = 3 * 24 * 60 * 60 * 1000;

export function usePulseChips(): PulseChipData[] {
  const { agents } = useLibraryAgents();

  const sitrepItems = useSitrepItems(agents, 5, THREE_DAYS_MS);

  return useMemo(() => {
    return sitrepItems.map((item) => ({
      id: item.id,
      agentID: item.agentID,
      name: item.agentName,
      status: item.status,
      priority: item.priority,
      shortMessage: item.message,
    }));
  }, [sitrepItems]);
}
