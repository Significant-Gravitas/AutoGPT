"use client";

import { cn } from "@/lib/utils";
import { Drawer } from "vaul";
import { useCopilotUIStore } from "../../store";
import { ArtifactContent } from "./components/ArtifactContent";
import { ArtifactMinimizedStrip } from "./components/ArtifactMinimizedStrip";
import { ArtifactPanelHeader } from "./components/ArtifactPanelHeader";
import { useArtifactPanel } from "./useArtifactPanel";

interface Props {
  mobile?: boolean;
}

export function ArtifactPanel({ mobile }: Props) {
  const {
    isMinimized,
    isMaximized,
    activeArtifact,
    history,
    isSourceView,
    classification,
    setIsSourceView,
    minimizeArtifactPanel,
    maximizeArtifactPanel,
    restoreArtifactPanel,
    goBackArtifact,
    canCopy,
    handleCopy,
    handleDownload,
  } = useArtifactPanel();
  const clearArtifactPreview = useCopilotUIStore((s) => s.clearArtifactPreview);

  if (!activeArtifact || !classification) return null;

  const headerProps = {
    artifact: activeArtifact,
    classification,
    canGoBack: history.length > 0,
    isMaximized,
    isSourceView,
    hasSourceToggle: classification.hasSourceToggle,
    mobile: !!mobile,
    canCopy,
    onBack: goBackArtifact,
    onClose: clearArtifactPreview,
    onMinimize: minimizeArtifactPanel,
    onMaximize: maximizeArtifactPanel,
    onRestore: restoreArtifactPanel,
    onCopy: handleCopy,
    onDownload: handleDownload,
    onSourceToggle: setIsSourceView,
  };

  if (isMinimized) {
    return (
      <div className="fixed right-0 top-[72px] z-[60] h-[calc(100vh-72px)]">
        <ArtifactMinimizedStrip
          artifact={activeArtifact}
          classification={classification}
          onExpand={restoreArtifactPanel}
        />
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
          className={cn(
            "fixed right-0 top-0 z-[70] flex h-full flex-col bg-white shadow-xl outline-none",
            mobile
              ? "w-full"
              : isMaximized
                ? "w-[85vw]"
                : "w-[640px] max-w-[90vw]",
          )}
          // Override vaul's `[data-vaul-drawer]{user-select:none}` rule so
          // artifact text is selectable.
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
