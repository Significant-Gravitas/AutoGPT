import { useAuth } from "@/lib/auth/hooks/useAuth";
import { trackFunnel } from "@/services/experts/experts-analytics";
import { useState } from "react";
import { readCollapsed, writeCollapsed } from "./helpers";

// Hide / Show for the suggested tasks: a per-user UI preference, read on
// mount and kept across reloads and sign-outs. Collapsing writes nothing to
// the tasks themselves.
export function useCollapsedSuggestions() {
  const { user } = useAuth();
  const userID = user?.id ?? null;
  const [choice, setChoice] = useState<{
    userID: string | null;
    collapsed: boolean;
  } | null>(null);
  const isCollapsed =
    choice && choice.userID === userID
      ? choice.collapsed
      : readCollapsed(userID);

  function toggle(openRows: number) {
    const next = !isCollapsed;
    setChoice({ userID, collapsed: next });
    writeCollapsed(userID, next);
    trackFunnel(
      next ? "home_suggestions_collapsed" : "home_suggestions_expanded",
      { open: openRows },
    );
  }

  return { isCollapsed, toggle };
}
