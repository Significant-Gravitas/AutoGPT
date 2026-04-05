"use client";

import { toast } from "@/components/molecules/Toast/use-toast";
import { useEffect, useState } from "react";
import { useCopilotUIStore } from "../../store";
import { downloadArtifact } from "./downloadArtifact";
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
      .then(() => {
        toast({ title: "Copied to clipboard" });
      })
      .catch(() => {
        toast({
          title: "Copy failed",
          description: "Couldn't read the file or access the clipboard.",
          variant: "destructive",
        });
      });
  }

  function handleDownload() {
    if (!activeArtifact) return;
    downloadArtifact(activeArtifact).catch(() => {
      toast({
        title: "Download failed",
        description: "Couldn't fetch the file.",
        variant: "destructive",
      });
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
