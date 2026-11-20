import { useQueryClient } from "@tanstack/react-query";
import type { ToolUIPart } from "ai";
import { useEffect, useRef } from "react";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { SESSION_LIST_QUERY_KEY } from "../../useSessionList";
import { appliedNewExpertCalls } from "./expertHandoff";

interface Args {
  parts: ToolUIPart[];
  isLive: boolean;
  readOnly: boolean;
}

/** Refreshes the roster and the chat list when a hire or a new expert lands,
 *  so the sidebar lists them, and their chat link finds them, without a
 *  reload. Only for confirms that land while this chat is open: one that was
 *  already applied when an old chat opened has nothing new to show. A held
 *  confirm lands later on an older message, so what counts is a call turning
 *  applied after mount, not the stream being live. */
export function useRefreshTeamOnApplied({ parts, isLive, readOnly }: Args) {
  const queryClient = useQueryClient();
  const seen = useRef<Set<string> | null>(null);
  const sawLive = useRef(false);
  const applied = appliedNewExpertCalls(parts).join(" ");

  useEffect(() => {
    if (isLive) sawLive.current = true;
    const ids = applied ? applied.split(" ") : [];
    if (seen.current === null) {
      seen.current = new Set(sawLive.current ? [] : ids);
    }
    const known = seen.current;
    const fresh = ids.filter((id) => !known.has(id));
    if (readOnly || fresh.length === 0) return;
    for (const id of fresh) known.add(id);
    void invalidateExpertRosterQueries(queryClient);
    void queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });
  }, [applied, isLive, readOnly, queryClient]);
}
