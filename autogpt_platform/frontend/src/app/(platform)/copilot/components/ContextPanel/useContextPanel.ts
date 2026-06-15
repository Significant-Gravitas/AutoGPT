"use client";

import { useCopilotUIStore } from "../../store";

export function useContextPanel() {
  const artifactPanel = useCopilotUIStore((s) => s.artifactPanel);
  const setActiveTab = useCopilotUIStore((s) => s.setActiveTab);
  const closeArtifactPanel = useCopilotUIStore((s) => s.closeArtifactPanel);
  const expandContextPanel = useCopilotUIStore((s) => s.expandContextPanel);
  const contextPanelWidth = useCopilotUIStore((s) => s.contextPanelWidth);
  const setContextPanelWidth = useCopilotUIStore((s) => s.setContextPanelWidth);

  const hasArtifact = artifactPanel.activeArtifact != null;
  const showRail = hasArtifact && artifactPanel.expandedPanel === "artifact";
  const showExpanded = !showRail && (artifactPanel.isOpen || hasArtifact);

  return {
    isOpen: artifactPanel.isOpen,
    activeTab: artifactPanel.activeTab,
    showRail,
    showExpanded,
    setActiveTab,
    closeArtifactPanel,
    expandContextPanel,
    contextPanelWidth,
    setContextPanelWidth,
  };
}
