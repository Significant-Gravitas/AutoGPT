"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import type { ArtifactPanelMode, ArtifactRef } from "../../../store";
import type { ArtifactClassification } from "../helpers";
import { PanelModeSwitch } from "./PanelModeSwitch";
import { SourceToggle } from "./SourceToggle";
import {
  ArrowLeft02Icon,
  Cancel01Icon,
  ComputerIcon,
  Copy01Icon,
  Download01Icon,
  Folder01Icon,
  ArrowExpandDiagonal01Icon,
  ArrowShrinkIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  /** Null while the panel shows only its computer face. */
  artifact: ArtifactRef | null;
  classification: ArtifactClassification | null;
  mode?: ArtifactPanelMode;
  /** Render the Artifact/Computer switch; the chat passes a session, the
   *  share viewer and tour do not. */
  showModeSwitch?: boolean;
  onModeChange?: (mode: ArtifactPanelMode) => void;
  canGoBack: boolean;
  isSourceView: boolean;
  hasSourceToggle: boolean;
  canCopy?: boolean;
  onBack: () => void;
  onClose: () => void;
  onCopy: () => void;
  onDownload: () => void;
  onOpenFiles: () => void;
  onSourceToggle: (isSource: boolean) => void;
  isFullscreen?: boolean;
  onToggleFullscreen?: () => void;
}

function HeaderButton({
  onClick,
  title,
  children,
}: {
  onClick: () => void;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          aria-label={title}
          className="flex size-8 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom">{title}</TooltipContent>
    </Tooltip>
  );
}

export function ArtifactPanelHeader({
  artifact,
  classification,
  canGoBack,
  isSourceView,
  hasSourceToggle,
  canCopy = true,
  onBack,
  onClose,
  onCopy,
  onDownload,
  onOpenFiles,
  onSourceToggle,
  isFullscreen = false,
  onToggleFullscreen,
  mode = "artifact",
  showModeSwitch = false,
  onModeChange,
}: Props) {
  const isComputer = mode === "computer" || artifact == null;
  const isFile = !isComputer && !artifact?.expert;
  const hasViewControls =
    (showModeSwitch && onModeChange) || (!isComputer && hasSourceToggle);

  return (
    <header className="shrink-0 border-b border-border bg-card">
      <div className="flex h-12 items-center gap-2 px-3 sm:px-4">
        {!isComputer && canGoBack && (
          <HeaderButton onClick={onBack} title="Back">
            <Icon icon={ArrowLeft02Icon} size={18} />
          </HeaderButton>
        )}
        <div className="flex min-w-0 flex-1 items-center gap-1.5 text-sm text-muted-foreground">
          {isComputer ? (
            <>
              <Icon icon={ComputerIcon} size={18} className="shrink-0" />
              <span className="truncate">Computer</span>
            </>
          ) : (
            <>
              <span className="truncate" title={artifact?.title}>
                {artifact?.title}
              </span>
              {classification && (
                <span className="shrink-0">· {classification.label}</span>
              )}
            </>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-0.5">
          {!isComputer && canCopy && (
            <HeaderButton onClick={onCopy} title="Copy">
              <Icon icon={Copy01Icon} size={18} />
            </HeaderButton>
          )}
          {isFile && (
            <>
              <HeaderButton onClick={onOpenFiles} title="All files">
                <Icon icon={Folder01Icon} size={18} />
              </HeaderButton>
              <HeaderButton onClick={onDownload} title="Download">
                <Icon icon={Download01Icon} size={18} />
              </HeaderButton>
            </>
          )}
          {onToggleFullscreen && (
            <HeaderButton
              onClick={onToggleFullscreen}
              title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            >
              <Icon
                icon={
                  isFullscreen ? ArrowShrinkIcon : ArrowExpandDiagonal01Icon
                }
                size={18}
              />
            </HeaderButton>
          )}
          <HeaderButton onClick={onClose} title="Close">
            <Icon icon={Cancel01Icon} size={18} />
          </HeaderButton>
        </div>
      </div>
      {hasViewControls && (
        <div className="flex flex-wrap items-center justify-between gap-2 px-3 pb-2 sm:px-4">
          {showModeSwitch && onModeChange && (
            <PanelModeSwitch
              mode={isComputer ? "computer" : "artifact"}
              hasArtifact={artifact != null}
              onChange={onModeChange}
            />
          )}
          {!isComputer && hasSourceToggle && (
            <SourceToggle
              isSourceView={isSourceView}
              onToggle={onSourceToggle}
            />
          )}
        </div>
      )}
    </header>
  );
}
