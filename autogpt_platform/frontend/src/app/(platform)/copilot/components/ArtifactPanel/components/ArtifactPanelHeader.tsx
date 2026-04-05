"use client";

import { cn } from "@/lib/utils";
import {
  ArrowLeft,
  ArrowsIn,
  ArrowsOut,
  Copy,
  DownloadSimple,
  Minus,
  X,
} from "@phosphor-icons/react";
import type { ArtifactRef } from "../../../store";
import type { ArtifactClassification } from "../helpers";
import { SourceToggle } from "./SourceToggle";

interface Props {
  artifact: ArtifactRef;
  classification: ArtifactClassification;
  canGoBack: boolean;
  isMaximized: boolean;
  isSourceView: boolean;
  hasSourceToggle: boolean;
  mobile?: boolean;
  canCopy?: boolean;
  onBack: () => void;
  onClose: () => void;
  onMinimize: () => void;
  onMaximize: () => void;
  onRestore: () => void;
  onCopy: () => void;
  onDownload: () => void;
  onSourceToggle: (isSource: boolean) => void;
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
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-label={title}
      className="rounded p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
    >
      {children}
    </button>
  );
}

export function ArtifactPanelHeader({
  artifact,
  classification,
  canGoBack,
  isMaximized,
  isSourceView,
  hasSourceToggle,
  mobile,
  canCopy = true,
  onBack,
  onClose,
  onMinimize,
  onMaximize,
  onRestore,
  onCopy,
  onDownload,
  onSourceToggle,
}: Props) {
  const Icon = classification.icon;

  return (
    <div className="sticky top-0 z-10 flex items-center gap-2 border-b border-zinc-200 bg-white px-3 py-2">
      {/* Left section */}
      <div className="flex min-w-0 flex-1 items-center gap-2">
        {canGoBack && (
          <HeaderButton onClick={onBack} title="Back">
            <ArrowLeft size={16} />
          </HeaderButton>
        )}
        <Icon size={16} className="shrink-0 text-zinc-400" />
        <span className="truncate text-sm font-medium text-zinc-900">
          {artifact.title}
        </span>
        <span
          className={cn(
            "shrink-0 rounded-full px-2 py-0.5 text-xs font-medium",
            artifact.origin === "user-upload"
              ? "bg-blue-50 text-blue-600"
              : "bg-violet-50 text-violet-600",
          )}
        >
          {classification.label}
        </span>
      </div>

      {/* Right section */}
      <div className="flex items-center gap-1">
        {hasSourceToggle && (
          <SourceToggle isSourceView={isSourceView} onToggle={onSourceToggle} />
        )}
        {canCopy && (
          <HeaderButton onClick={onCopy} title="Copy">
            <Copy size={16} />
          </HeaderButton>
        )}
        <HeaderButton onClick={onDownload} title="Download">
          <DownloadSimple size={16} />
        </HeaderButton>
        {!mobile && (
          <>
            <HeaderButton onClick={onMinimize} title="Minimize">
              <Minus size={16} />
            </HeaderButton>
            {isMaximized ? (
              <HeaderButton onClick={onRestore} title="Restore">
                <ArrowsIn size={16} />
              </HeaderButton>
            ) : (
              <HeaderButton onClick={onMaximize} title="Maximize">
                <ArrowsOut size={16} />
              </HeaderButton>
            )}
          </>
        )}
        <HeaderButton onClick={onClose} title="Close">
          <X size={16} />
        </HeaderButton>
      </div>
    </div>
  );
}
