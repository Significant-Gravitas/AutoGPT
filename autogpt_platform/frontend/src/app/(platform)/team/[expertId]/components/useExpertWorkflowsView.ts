import { Key, storage } from "@/services/storage/local-storage";
import { useEffect, useState } from "react";

export type WorkflowsView = "grid" | "list";

const DEFAULT_VIEW: WorkflowsView = "list";

export function useExpertWorkflowsView() {
  const [view, setViewState] = useState<WorkflowsView>(DEFAULT_VIEW);

  useEffect(() => {
    const stored = storage.get(Key.TEAM_WORKFLOWS_VIEW);
    if (stored === "grid" || stored === "list") setViewState(stored);
  }, []);

  function setView(next: WorkflowsView) {
    setViewState(next);
    storage.set(Key.TEAM_WORKFLOWS_VIEW, next);
  }

  return { view, setView };
}
