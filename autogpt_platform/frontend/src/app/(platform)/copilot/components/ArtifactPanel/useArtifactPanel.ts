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

  // Track viewport width reactively for maximize mode
  const [viewportWidth, setViewportWidth] = useState(
    typeof window !== "undefined" ? window.innerWidth : 1280,
  );
  useEffect(() => {
    function handleResize() {
      setViewportWidth(window.innerWidth);
    }
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const canCopy =
    classification != null &&
    classification.type !== "image" &&
    classification.type !== "download-only" &&
    classification.type !== "pdf";

  function handleCopy() {
    if (!activeArtifact || !canCopy) return;
    fetch(activeArtifact.sourceUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`Copy failed: ${res.status}`);
        return res.text();
      })
      .then((text) => navigator.clipboard.writeText(text))
      .catch(() => {
        /* clipboard permission denied or fetch failed — silent for now */
      });
  }

  function handleDownload() {
    if (!activeArtifact) return;
    // Fetch + blob URL so the `download` attribute is honored even when the
    // source URL is cross-origin (GCS signed URLs) and across browsers that
    // require the anchor to be in the DOM (Firefox).
    const safeName =
      activeArtifact.title.replace(/[\\/:*?"<>|\x00-\x1f]/g, "_") || "download";
    fetch(activeArtifact.sourceUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`Download failed: ${res.status}`);
        return res.blob();
      })
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = safeName;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      })
      .catch(() => {
        /* fetch or blob creation failed — silent for now */
      });
  }

  // Always clamp against the current viewport so a previously-dragged-wide
  // panel doesn't spill offscreen after the user resizes their window.
  const maxWidth = viewportWidth * 0.85;
  const effectiveWidth = artifactPanel.isMaximized
    ? maxWidth
    : Math.min(artifactPanel.width, maxWidth);

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
    canCopy,
    handleCopy,
    handleDownload,
  };
}
