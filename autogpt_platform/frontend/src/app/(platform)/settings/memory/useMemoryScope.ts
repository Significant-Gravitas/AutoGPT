import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useEffect, useState } from "react";
import {
  getActiveExperts,
  getScopeName,
  readExpertScopeFromUrl,
} from "./helpers";

export function useMemoryScope() {
  // The Soul drawer's learned-notes panel links here with ?expert=<id>, so
  // the link opens on that expert. Read once — the dropdown owns it after.
  const [scopeExpertID, setScopeExpertID] = useState<string | null>(
    readExpertScopeFromUrl,
  );

  const expertsQuery = useListExperts();
  const expertsSettled = expertsQuery.data?.status === 200;
  const experts = getActiveExperts(
    expertsQuery.data?.status === 200 ? expertsQuery.data.data : undefined,
  );

  // A selected expert can disappear (fired in another tab, roster changed) —
  // fall back to Otto instead of driving queries with a dead scope id.
  useEffect(() => {
    if (!scopeExpertID || !expertsSettled) return;
    if (!experts.some((expert) => expert.id === scopeExpertID)) {
      setScopeExpertID(null);
    }
  }, [scopeExpertID, expertsSettled, experts]);

  return {
    scopeExpertID,
    selectScope: setScopeExpertID,
    experts,
    scopeName: getScopeName(scopeExpertID, experts),
  };
}
