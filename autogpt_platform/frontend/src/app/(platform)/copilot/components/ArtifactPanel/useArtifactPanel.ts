"use client";

import { toast } from "@/components/molecules/Toast/use-toast";
import { useEffect, useState } from "react";
import { useCopilotUIStore } from "../../store";
import { getCachedArtifactContent } from "./components/useArtifactContent";
import { downloadArtifact } from "./downloadArtifact";
import { classifyArtifact } from "./helpers";

export function useArtifactPanel() {
  const artifactPanel = useCopilotUIStore((s) => s.artifactPanel);
  const clearArtifactPreview = useCopilotUIStore((s) => s.clearArtifactPreview);
  const goBackArtifact = useCopilotUIStore((s) => s.goBackArtifact);
  const showFilesTab = useCopilotUIStore((s) => s.showFilesTab);
  const artifactPanelWidth = useCopilotUIStore((s) => s.artifactPanelWidth);
  const setArtifactPanelWidth = useCopilotUIStore(
    (s) => s.setArtifactPanelWidth,
  );

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

  // Escape-to-close is owned by the vaul Drawer.Root in ArtifactPanel — its
  // onOpenChange already routes to clearArtifactPreview. A manual document
  // listener here would self-block on the drawer's own [role="dialog"].

  const canCopy =
    classification != null &&
    classification.type !== "image" &&
    classification.type !== "video" &&
    classification.type !== "download-only" &&
    classification.type !== "pdf";

  function handleCopy() {
    if (!activeArtifact || !canCopy) return;
    // Reuse content already fetched by the preview pane when available —
    // Copy should feel instant, not trigger a second network round-trip.
    const cached = getCachedArtifactContent(activeArtifact.id);
    const textPromise = cached
      ? Promise.resolve(cached)
      : fetch(activeArtifact.sourceUrl).then((res) => {
          if (!res.ok) throw new Error(`Copy failed: ${res.status}`);
          return res.text();
        });
    textPromise
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

  return {
    activeArtifact,
    history: artifactPanel.history,
    isSourceView,
    classification,
    setIsSourceView,
    clearArtifactPreview,
    goBackArtifact,
    showFilesTab,
    canCopy,
    handleCopy,
    handleDownload,
    artifactPanelWidth,
    setArtifactPanelWidth,
  };
}
