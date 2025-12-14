import { useEffect, useMemo } from "react";
import { buildAgentInfoMap, useLibraryAgentsStore } from "./store";

let initialized = false;

export function useLibraryAgents() {
  const { agents, isRefreshing, lastUpdatedAt, loadFromCache, refreshAll } =
    useLibraryAgentsStore();

  useEffect(() => {
    if (!initialized) {
      loadFromCache();
      void refreshAll();
      initialized = true;
    }
  }, [loadFromCache, refreshAll]);

  const agentInfoMap = useMemo(() => buildAgentInfoMap(agents), [agents]);

  return { agents, agentInfoMap, isRefreshing, lastUpdatedAt };
}
