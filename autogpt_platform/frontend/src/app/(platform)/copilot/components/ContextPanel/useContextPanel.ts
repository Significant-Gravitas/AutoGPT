"use client";

import { useCopilotUIStore } from "../../store";

export function useContextPanel() {
  const artifactPanel = useCopilotUIStore((s) => s.artifactPanel);
  const setActiveTab = useCopilotUIStore((s) => s.setActiveTab);
  const closeArtifactPanel = useCopilotUIStore((s) => s.closeArtifactPanel);
  const contextPanelWidth = useCopilotUIStore((s) => s.contextPanelWidth);
  const setContextPanelWidth = useCopilotUIStore((s) => s.setContextPanelWidth);

  // The artifact takes over the right region while it's open, so the Context
  // Panel is hidden then (you return to it by closing the artifact).
  const hasArtifact = artifactPanel.activeArtifact != null;
  const showExpanded = artifactPanel.isOpen && !hasArtifact;

  return {
    isOpen: artifactPanel.isOpen,
    activeTab: artifactPanel.activeTab,
    showExpanded,
    setActiveTab,
    closeArtifactPanel,
    contextPanelWidth,
    setContextPanelWidth,
  };
}
