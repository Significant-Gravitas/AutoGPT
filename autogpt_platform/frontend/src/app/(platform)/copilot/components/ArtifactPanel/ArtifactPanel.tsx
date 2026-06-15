"use client";

import { Drawer } from "vaul";
import { MIN_ARTIFACT_PANEL_WIDTH } from "../../store";
import { PanelResizeHandle } from "../PanelResizeHandle";
import { ArtifactContent } from "./components/ArtifactContent";
import { ArtifactPanelHeader } from "./components/ArtifactPanelHeader";
import { ArtifactRail } from "./components/ArtifactRail";
import { useArtifactPanel } from "./useArtifactPanel";

interface Props {
  mobile?: boolean;
}

export function ArtifactPanel({ mobile }: Props) {
  const {
    activeArtifact,
    expandedPanel,
    history,
    isSourceView,
    classification,
    setIsSourceView,
    clearArtifactPreview,
    expandArtifactPanel,
    goBackArtifact,
    canCopy,
    handleCopy,
    handleDownload,
    artifactPanelWidth,
    setArtifactPanelWidth,
  } = useArtifactPanel();

  if (!activeArtifact || !classification) return null;

  const headerProps = {
    artifact: activeArtifact,
    classification,
    canGoBack: history.length > 0,
    isSourceView,
    hasSourceToggle: classification.hasSourceToggle,
    canCopy,
    onBack: goBackArtifact,
    onClose: clearArtifactPreview,
    onCopy: handleCopy,
    onDownload: handleDownload,
    onSourceToggle: setIsSourceView,
  };

  if (!mobile) {
    if (expandedPanel === "context") {
      return (
        <ArtifactRail
          artifact={activeArtifact}
          onExpand={expandArtifactPanel}
        />
      );
    }
    return (
      <div
        data-artifact-panel
        style={{ width: artifactPanelWidth, userSelect: "text" }}
        className="relative flex h-full shrink-0 flex-col border-l border-l-[#80808017] bg-sidebar"
      >
        <PanelResizeHandle
          panelSelector="[data-artifact-panel]"
          onWidthChange={setArtifactPanelWidth}
          minWidth={MIN_ARTIFACT_PANEL_WIDTH}
        />
        <div className="flex min-h-0 w-full flex-1 flex-col overflow-hidden">
          <ArtifactPanelHeader {...headerProps} />
          <ArtifactContent
            artifact={activeArtifact}
            isSourceView={isSourceView}
            classification={classification}
          />
        </div>
      </div>
    );
  }

  return (
    <Drawer.Root
      open={!!activeArtifact}
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
            {activeArtifact.title}
          </Drawer.Title>
          <ArtifactPanelHeader {...headerProps} />
          <ArtifactContent
            artifact={activeArtifact}
            isSourceView={isSourceView}
            classification={classification}
          />
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
