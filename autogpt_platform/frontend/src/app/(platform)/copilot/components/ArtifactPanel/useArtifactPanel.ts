"use client";

import { toast } from "@/components/molecules/Toast/use-toast";
import { useEffect, useState } from "react";
import { useCopilotUIStore } from "../../store";
import { getCachedArtifactContent } from "./components/useArtifactContent";
import { downloadArtifact } from "./downloadArtifact";
import { classifyArtifact } from "./helpers";

// SSR fallback for viewport width before window is available.
const DEFAULT_VIEWPORT_WIDTH = 1280;

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
        if (document.querySelector('[role="dialog"], [data-state="open"]'))
          return;
        closeArtifactPanel();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [artifactPanel.isOpen, closeArtifactPanel]);

  // Track viewport width reactively for maximize mode.
  const [viewportWidth, setViewportWidth] = useState(
    typeof window !== "undefined" ? window.innerWidth : DEFAULT_VIEWPORT_WIDTH,
  );
  useEffect(() => {
    // Throttle to ~10Hz: resize fires continuously during drag, but we only
    // need the panel width to follow the viewport within a frame or two.
    let timer: ReturnType<typeof setTimeout> | null = null;
    function handleResize() {
      if (timer) return;
      timer = setTimeout(() => {
        setViewportWidth(window.innerWidth);
        timer = null;
      }, 100);
    }
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      if (timer) clearTimeout(timer);
    };
  }, []);

  const canCopy =
    classification != null &&
    classification.type !== "image" &&
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
