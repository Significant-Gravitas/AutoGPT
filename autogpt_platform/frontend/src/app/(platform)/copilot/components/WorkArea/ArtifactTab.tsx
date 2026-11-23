"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  ArrowLeft01Icon,
  Download01Icon,
  FullScreenIcon,
} from "@hugeicons/core-free-icons";
import { useState } from "react";
import { ArtifactPreview } from "../ArtifactPanel/components/ArtifactPreview";
import { useArtifactFullscreen } from "../ArtifactPanel/useArtifactFullscreen";
import { useArtifactPanel } from "../ArtifactPanel/useArtifactPanel";
import { FOCUS } from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { CopyTool, Tool, ViewSwitch } from "./artifactTools";
import { DocumentReader } from "./DocumentReader";

const PANEL_PROSE = "px-3 pt-2 pb-10";

/** The open file: a slim row of tools over the viewer. Its name is on its
 *  tab, so the row carries only what you can do with it. A written document
 *  lies straight on the panel in the chat's own type, and "Read" lifts it
 *  over the chat; anything else keeps a page under it. */
export function ArtifactTab() {
  const panel = useArtifactPanel();
  const { fullscreenRef, isFullscreen, canFullscreen, toggleFullscreen } =
    useArtifactFullscreen();
  const [isReading, setIsReading] = useState(false);
  const { activeArtifact, classification } = panel;
  if (!activeArtifact || !classification) return null;

  const isDocument =
    classification.type === "markdown" || classification.type === "text";

  return (
    <div
      ref={fullscreenRef}
      data-artifact-panel
      className={cn(
        "flex h-full min-h-0 flex-col",
        !isDocument &&
          "overflow-hidden rounded-card bg-surface shadow-(--inbox-shadow-card)",
      )}
    >
      <div
        className={cn(
          "flex h-9 shrink-0 items-center gap-1 px-1",
          !isDocument && "shadow-(--inbox-shadow-hairline-b)",
        )}
      >
        {panel.history.length > 0 ? (
          <Tool
            icon={ArrowLeft01Icon}
            label="Back"
            onClick={panel.goBackArtifact}
          />
        ) : null}
        {classification.hasSourceToggle ? (
          <ViewSwitch
            isSource={panel.isSourceView}
            onChange={panel.setIsSourceView}
          />
        ) : null}
        <span className="flex-1" />
        {panel.canCopy ? <CopyTool onCopy={panel.handleCopy} /> : null}
        <Tool
          icon={Download01Icon}
          label="Download"
          onClick={panel.handleDownload}
        />
        {isDocument ? (
          <button
            type="button"
            onClick={() => setIsReading(true)}
            className={cn(
              "ml-1 flex h-7 shrink-0 items-center gap-1.5 rounded-md bg-foreground-ink px-2.5 text-body-sm font-medium text-surface transition-transform duration-150 ease-out active:scale-95 motion-reduce:transition-none",
              FOCUS,
            )}
          >
            <Icon icon={FullScreenIcon} size={13} tone="inherit" />
            Read
          </button>
        ) : canFullscreen ? (
          <Tool
            icon={FullScreenIcon}
            label={isFullscreen ? "Leave full screen" : "Full screen"}
            onClick={() => void toggleFullscreen()}
          />
        ) : null}
      </div>
      <ArtifactPreview
        artifact={activeArtifact}
        isSourceView={panel.isSourceView}
        classification={classification}
        isBare={isDocument}
        prose={isDocument ? PANEL_PROSE : undefined}
      />
      {isReading && isDocument ? (
        <DocumentReader
          artifact={activeArtifact}
          classification={classification}
          isSourceView={panel.isSourceView}
          onSourceView={panel.setIsSourceView}
          canCopy={panel.canCopy}
          onCopy={panel.handleCopy}
          onDownload={panel.handleDownload}
          onClose={() => setIsReading(false)}
        />
      ) : null}
    </div>
  );
}
