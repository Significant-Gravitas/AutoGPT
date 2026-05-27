"use client";

import { useCopilotUIStore } from "../../store";

export function useContextPanel() {
  const artifactPanel = useCopilotUIStore((s) => s.artifactPanel);
  const setActiveTab = useCopilotUIStore((s) => s.setActiveTab);
  const setArtifactPanelWidth = useCopilotUIStore(
    (s) => s.setArtifactPanelWidth,
  );
  const closeArtifactPanel = useCopilotUIStore((s) => s.closeArtifactPanel);

  const view = artifactPanel.activeArtifact ? "preview" : "tabs";

  return {
    isOpen: artifactPanel.isOpen,
    activeTab: artifactPanel.activeTab,
    view,
    setActiveTab,
    setArtifactPanelWidth,
    closeArtifactPanel,
  };
}
