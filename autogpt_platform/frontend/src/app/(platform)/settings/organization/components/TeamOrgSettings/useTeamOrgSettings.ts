"use client";

import { useSearchParams } from "next/navigation";
import { useState } from "react";

import { getAllowedTabs, resolveTab } from "./helpers";

interface Args {
  isAdmin: boolean;
  canManageBilling: boolean;
}

export function useTeamOrgSettings({ isAdmin, canManageBilling }: Args) {
  const searchParams = useSearchParams();
  const [requestedTab, setRequestedTab] = useState<string | null>(() =>
    searchParams.get("tab"),
  );
  const allowedTabs = getAllowedTabs({ isAdmin, canManageBilling });

  return {
    activeTab: resolveTab(requestedTab, allowedTabs),
    canSeeInvitations: allowedTabs.includes("invitations"),
    canSeeBilling: allowedTabs.includes("billing"),
    handleTabChange: setRequestedTab,
  };
}
