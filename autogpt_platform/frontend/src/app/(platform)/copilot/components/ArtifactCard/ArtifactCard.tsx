"use client";

import { useStableCallback } from "@/hooks/useStableCallback";
import { toast } from "@/components/molecules/Toast/use-toast";
import { cn } from "@/lib/utils";
import { useEffect } from "react";
import type { ArtifactRef } from "../../store";
import { useCopilotUIStore } from "../../store";
import { downloadArtifact } from "../ArtifactPanel/downloadArtifact";
import { classifyArtifact } from "../ArtifactPanel/helpers";
import { ArrowRight01Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  artifact: ArtifactRef;
  /** Read-only mode: opt out of the side-effects that only make sense
   *  for the owner (auto-registering with the artifact panel so a
   *  newly arriving file can pop the panel open).
   *
   *  Click behaviour is unchanged from the owner case: an openable
   *  artifact still calls ``openArtifact`` and renders inside
   *  ``ArtifactPanel`` (the public share viewer mounts the panel too),
   *  while a non-openable asset falls back to a direct download.  In
   *  other words, ``readOnly`` does NOT force every click to download
   *  — that's only true for non-openable assets, regardless of mode. */
  readOnly?: boolean;
}

function formatSize(bytes?: number): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function ArtifactCard({ artifact, readOnly }: Props) {
  const isActive = useCopilotUIStore(
    (s) => s.artifactPanel.activeArtifact?.id === artifact.id,
  );
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const registerArtifactForAutoOpen = useCopilotUIStore(
    (s) => s.registerArtifactForAutoOpen,
  );

  // Register this artifact on mount — the store decides whether to auto-open.
  // Fires once per artifact ID; subsequent renders with the same ID are no-ops
  // in the store (knownIds check).
  // Skipped in readOnly mode — the share viewer has no panel for auto-open
  // to target, and registering would just pollute the store.
  const registerForAutoOpen = useStableCallback(() => {
    registerArtifactForAutoOpen(artifact);
  });
  useEffect(() => {
    if (readOnly) return;
    registerForAutoOpen();
  }, [artifact.id, artifact.mimeType, registerArtifactForAutoOpen, readOnly]);

  const classification = classifyArtifact(
    artifact.mimeType,
    artifact.title,
    artifact.sizeBytes,
  );

  function handleDownloadOnly() {
    downloadArtifact(artifact).catch(() => {
      toast({
        title: "Download failed",
        description: "Couldn't fetch the file.",
        variant: "destructive",
      });
    });
  }

  if (!classification.openable) {
    return (
      <button
        type="button"
        onClick={handleDownloadOnly}
        // oxlint-disable-next-line design-system/no-arbitrary-values -- duration-500 would also slow the hover colour transition
        className="my-1 flex w-full min-w-0 items-center gap-3 rounded-2xl border border-border-base bg-surface px-3 py-2.5 text-left transition-colors animate-in [animation-duration:500ms] fade-in fill-mode-both slide-in-from-bottom-2 hover:bg-surface-muted"
      >
        <Icon
          icon={classification.icon}
          size={20}
          className="shrink-0 text-foreground-subtle"
        />
        <div className="min-w-0 flex-1">
          <Text variant="body-medium-tight" tone="primary" className="truncate">
            {artifact.title}
          </Text>
          <Text variant="small-tight" tone="subtle">
            {classification.label}
            {artifact.sizeBytes
              ? ` \u2022 ${formatSize(artifact.sizeBytes)}`
              : ""}
          </Text>
        </div>
        <Icon
          icon={Download01Icon}
          size={16}
          className="shrink-0 text-foreground-subtle"
        />
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={() => openArtifact(artifact)}
      className={cn(
        // oxlint-disable-next-line design-system/no-arbitrary-values -- duration-500 would also slow the hover colour transition
        "my-1 flex w-full min-w-0 items-center gap-3 rounded-2xl border bg-surface px-3 py-2.5 text-left transition-colors animate-in [animation-duration:500ms] fade-in fill-mode-both slide-in-from-bottom-2 hover:bg-surface-muted",
        isActive
          ? "border-brand/40 bg-accent-surface/50"
          : "border-border-base",
      )}
    >
      <Icon
        icon={classification.icon}
        size={20}
        className={cn(
          "shrink-0",
          isActive ? "text-brand" : "text-foreground-subtle",
        )}
      />
      <div className="min-w-0 flex-1">
        <Text variant="body-medium-tight" tone="primary" className="truncate">
          {artifact.title}
        </Text>
        <Text variant="small-tight" tone="subtle">
          <span
            className={cn(
              "inline-block rounded-full px-1.5 py-0.5 text-xs font-medium",
              artifact.origin === "user-upload"
                ? "bg-info-surface text-info"
                : "bg-accent-surface text-brand",
            )}
          >
            {classification.label}
          </span>
          {artifact.sizeBytes
            ? ` \u2022 ${formatSize(artifact.sizeBytes)}`
            : ""}
        </Text>
      </div>
      <Icon
        icon={ArrowRight01Icon}
        size={16}
        className={cn(
          "shrink-0",
          isActive ? "text-brand/60" : "text-foreground-disabled",
        )}
      />
    </button>
  );
}
