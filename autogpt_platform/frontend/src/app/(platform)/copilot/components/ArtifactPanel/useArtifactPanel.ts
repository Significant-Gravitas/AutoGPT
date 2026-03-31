"use client";

import { useEffect, useState } from "react";
import { useCopilotUIStore } from "../../store";
import { classifyArtifact } from "./helpers";

export function useArtifactPanel() {
  const artifactPanel = useCopilotUIStore((s) => s.artifactPanel);
  const closeArtifactPanel = useCopilotUIStore((s) => s.closeArtifactPanel);
  const minimizeArtifactPanel = useCopilotUIStore(
    (s) => s.minimizeArtifactPanel,
  );
  const maximizeArtifactPanel = useCopilotUIStore(
    (s) => s.maximizeArtifactPanel,
  );
  const restoreArtifactPanel = useCopilotUIStore((s) => s.restoreArtifactPanel);
  const setArtifactPanelWidth = useCopilotUIStore(
    (s) => s.setArtifactPanelWidth,
  );
  const goBackArtifact = useCopilotUIStore((s) => s.goBackArtifact);

  const [isSourceView, setIsSourceView] = useState(false);

  const { activeArtifact } = artifactPanel;

  const classification = activeArtifact
    ? classifyArtifact(
        activeArtifact.mimeType,
        activeArtifact.title,
        activeArtifact.sizeBytes,
      )
    : null;

  // Reset source view when switching artifacts
  useEffect(() => {
    setIsSourceView(false);
  }, [activeArtifact?.id]);

  // Keyboard: Escape to close
  useEffect(() => {
    if (!artifactPanel.isOpen) return;

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        closeArtifactPanel();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [artifactPanel.isOpen, closeArtifactPanel]);

  function handleCopy() {
    if (!activeArtifact) return;
    fetch(activeArtifact.sourceUrl)
      .then((res) => res.text())
      .then((text) => navigator.clipboard.writeText(text));
  }

  function handleDownload() {
    if (!activeArtifact) return;
    const a = document.createElement("a");
    a.href = activeArtifact.sourceUrl;
    a.download = activeArtifact.title;
    a.click();
  }

  // Compute effective width
  const effectiveWidth = artifactPanel.isMaximized
    ? window.innerWidth * 0.85
    : artifactPanel.width;

  return {
    ...artifactPanel,
    effectiveWidth,
    isSourceView,
    classification,
    setIsSourceView,
    closeArtifactPanel,
    minimizeArtifactPanel,
    maximizeArtifactPanel,
    restoreArtifactPanel,
    setArtifactPanelWidth,
    goBackArtifact,
    handleCopy,
    handleDownload,
  };
}
