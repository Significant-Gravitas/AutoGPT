"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Drawer } from "vaul";
import { MIN_ARTIFACT_PANEL_WIDTH, PANEL_RESERVED_WIDTH } from "../../store";
import { PanelResizeHandle } from "../PanelResizeHandle";
import { ArtifactContent } from "./components/ArtifactContent";
import { ArtifactPanelHeader } from "./components/ArtifactPanelHeader";
import { useArtifactPanel } from "./useArtifactPanel";

// Matched to the context panel so the two side rails move as one system.
const PANEL_EASE: [number, number, number, number] = [0.32, 0.72, 0, 1];
const PANEL_DURATION = 0.3;

interface Props {
  mobile?: boolean;
  /** The desktop copilot chat renders its own sidebar-right close control;
   *  standalone hosts (share viewer, tour, mobile drawer) do not, so their
   *  header keeps its Close button. */
  hasExternalClose?: boolean;
}

export function ArtifactPanel({ mobile, hasExternalClose }: Props) {
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

  // Hold the last live artifact so both the mobile drawer and the desktop
  // panel can keep rendering its contents while the close animation plays —
  // by then `activeArtifact` is already null, and unmounting outright would
  // snap the panel shut without animating.
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
  const [isResizing, setIsResizing] = useState(false);
  const shouldReduceMotion = useReducedMotion();
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

  // Keep painting the outgoing artifact through the close tween — by then
  // `activeArtifact` is already null, same reason the mobile drawer holds it.
  const shown = lastShownRef.current;

  // jsdom reports offsetWidth 0 — treat non-positive readings as "unknown"
  // and fall back to the stored width.
  const renderedWidth =
    availableWidth == null || availableWidth <= 0
      ? artifactPanelWidth
      : Math.min(artifactPanelWidth, availableWidth);

  // Width is the animated property because the panel pushes the chat column
  // rather than overlaying it. Dragging the handle bypasses the tween — a
  // queued 300ms tween per pointer move would trail the cursor.
  const transition =
    shouldReduceMotion || isResizing
      ? { duration: 0 }
      : { duration: PANEL_DURATION, ease: PANEL_EASE };

  return (
    <AnimatePresence initial={false}>
      {showDesktopPanel && shown && (
        <motion.div
          ref={panelRef}
          data-artifact-panel
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: renderedWidth, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={transition}
          style={{ userSelect: "text" }}
          className="relative h-full shrink-0 border-l border-l-[#80808017] bg-sidebar"
        >
          {/* Sibling of the clip, not a child of it: the handle straddles the
              border, so clipping it here would halve the drag target. */}
          <PanelResizeHandle
            panelSelector="[data-artifact-panel]"
            onWidthChange={setArtifactPanelWidth}
            onResizingChange={setIsResizing}
            minWidth={MIN_ARTIFACT_PANEL_WIDTH}
          />
          <div className="h-full overflow-hidden">
            {/* Fixed inner width so the header and content keep their final
                layout while the shell widens — nothing reflows mid-tween. */}
            <div
              style={{ width: renderedWidth }}
              className="flex h-full min-h-0 flex-col overflow-hidden"
            >
              <ArtifactPanelHeader
                artifact={shown.artifact}
                classification={shown.classification}
                hasExternalClose={hasExternalClose}
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
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
