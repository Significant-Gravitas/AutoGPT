"use client";

import { useEffect, useRef, useState } from "react";
import { Drawer } from "vaul";
import { MIN_ARTIFACT_PANEL_WIDTH, PANEL_RESERVED_WIDTH } from "../../store";
import { PanelResizeHandle } from "../PanelResizeHandle";
import { ArtifactContent } from "./components/ArtifactContent";
import { ArtifactPanelHeader } from "./components/ArtifactPanelHeader";
import { useArtifactPanel } from "./useArtifactPanel";

interface Props {
  mobile?: boolean;
}

export function ArtifactPanel({ mobile }: Props) {
  const {
    activeArtifact,
    history,
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
  } = useArtifactPanel();

  // Hold the last live artifact so the mobile drawer can keep rendering its
  // contents while vaul plays the slide-out animation — by then
  // `activeArtifact` is already null, and unmounting the whole drawer would
  // snap it shut without animating. Desktop returns null immediately (no exit
  // animation expected there).
  const lastShownRef = useRef<{
    artifact: NonNullable<typeof activeArtifact>;
    classification: NonNullable<typeof classification>;
  } | null>(null);
  if (activeArtifact && classification) {
    lastShownRef.current = { artifact: activeArtifact, classification };
  }

  // The stored width is a preference, not a guarantee: the panel is shrink-0,
  // so on narrow viewports it would overflow the flex row and get clipped on
  // the right. Clamp the rendered width to the space the row actually leaves
  // (same reservation the resize handle uses while dragging); the stored
  // width is left untouched and applies again once the viewport grows.
  const panelRef = useRef<HTMLDivElement>(null);
  const [availableWidth, setAvailableWidth] = useState<number | null>(null);
  const showDesktopPanel = !mobile && !!activeArtifact && !!classification;
  useEffect(() => {
    if (!showDesktopPanel || typeof ResizeObserver === "undefined") return;
    const parent = panelRef.current?.parentElement;
    if (!parent) return;
    const update = () =>
      setAvailableWidth(parent.offsetWidth - PANEL_RESERVED_WIDTH);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(parent);
    return () => {
      observer.disconnect();
      // Drop the measurement on close so a reopen after a viewport resize
      // never renders a frame with a stale width.
      setAvailableWidth(null);
    };
  }, [showDesktopPanel]);

  if (mobile) {
    const shown = lastShownRef.current;
    if (!shown) return null;

    return (
      <Drawer.Root
        open={!!activeArtifact && !!classification}
        onOpenChange={(open) => !open && clearArtifactPreview()}
        direction="right"
        handleOnly
        noBodyStyles
        modal={false}
      >
        <Drawer.Portal>
          {/* Manual backdrop — vaul's Drawer.Overlay wraps RemoveScroll, which
              adds padding-right to compensate for scrollbar removal. Our layout
              scrolls internally (no body scrollbar), so that padding visibly
              shifts the underlying page. modal={false} disables RemoveScroll;
              we render our own backdrop with click-to-close. */}
          <div
            onClick={clearArtifactPreview}
            className="fixed inset-0 z-[60] bg-black/20 backdrop-blur-[2px]"
            aria-hidden="true"
          />
          <Drawer.Content
            className="fixed right-0 top-0 z-[70] flex h-full w-full flex-col bg-white shadow-xl outline-none"
            style={{ userSelect: "text" }}
            aria-describedby={undefined}
          >
            <Drawer.Title className="sr-only">
              {shown.artifact.title}
            </Drawer.Title>
            <ArtifactPanelHeader
              artifact={shown.artifact}
              classification={shown.classification}
              canGoBack={history.length > 0}
              isSourceView={isSourceView}
              hasSourceToggle={shown.classification.hasSourceToggle}
              canCopy={canCopy}
              onBack={goBackArtifact}
              onClose={clearArtifactPreview}
              onCopy={handleCopy}
              onDownload={handleDownload}
              onOpenFiles={showFilesTab}
              onSourceToggle={setIsSourceView}
            />
            <ArtifactContent
              artifact={shown.artifact}
              isSourceView={isSourceView}
              classification={shown.classification}
            />
          </Drawer.Content>
        </Drawer.Portal>
      </Drawer.Root>
    );
  }

  if (!activeArtifact || !classification) return null;

  // jsdom reports offsetWidth 0 — treat non-positive readings as "unknown"
  // and fall back to the stored width.
  const renderedWidth =
    availableWidth == null || availableWidth <= 0
      ? artifactPanelWidth
      : Math.min(artifactPanelWidth, availableWidth);

  return (
    <div
      ref={panelRef}
      data-artifact-panel
      style={{ width: renderedWidth, userSelect: "text" }}
      className="relative flex h-full shrink-0 flex-col border-l border-l-[#80808017] bg-sidebar"
    >
      <PanelResizeHandle
        panelSelector="[data-artifact-panel]"
        onWidthChange={setArtifactPanelWidth}
        minWidth={MIN_ARTIFACT_PANEL_WIDTH}
      />
      <div className="flex min-h-0 w-full flex-1 flex-col overflow-hidden">
        <ArtifactPanelHeader
          artifact={activeArtifact}
          classification={classification}
          canGoBack={history.length > 0}
          isSourceView={isSourceView}
          hasSourceToggle={classification.hasSourceToggle}
          canCopy={canCopy}
          onBack={goBackArtifact}
          onClose={clearArtifactPreview}
          onCopy={handleCopy}
          onDownload={handleDownload}
          onOpenFiles={showFilesTab}
          onSourceToggle={setIsSourceView}
        />
        <ArtifactContent
          artifact={activeArtifact}
          isSourceView={isSourceView}
          classification={classification}
        />
      </div>
    </div>
  );
}
