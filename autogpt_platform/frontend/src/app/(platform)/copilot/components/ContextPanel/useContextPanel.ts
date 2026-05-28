"use client";

import { useCopilotUIStore } from "../../store";

export function useContextPanel() {
  const artifactPanel = useCopilotUIStore((s) => s.artifactPanel);
  const setActiveTab = useCopilotUIStore((s) => s.setActiveTab);
  const setArtifactPanelWidth = useCopilotUIStore(
    (s) => s.setArtifactPanelWidth,
  );
  const closeArtifactPanel = useCopilotUIStore((s) => s.closeArtifactPanel);

  return {
    isOpen: artifactPanel.isOpen,
    activeTab: artifactPanel.activeTab,
    width: artifactPanel.width,
    setActiveTab,
    setArtifactPanelWidth,
    closeArtifactPanel,
  };
}
