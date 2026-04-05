"use client";

import { toast } from "@/components/molecules/Toast/use-toast";
import { cn } from "@/lib/utils";
import { CaretRight, DownloadSimple } from "@phosphor-icons/react";
import type { ArtifactRef } from "../../store";
import { useCopilotUIStore } from "../../store";
import { downloadArtifact } from "../ArtifactPanel/downloadArtifact";
import { classifyArtifact } from "../ArtifactPanel/helpers";

interface Props {
  artifact: ArtifactRef;
}

function formatSize(bytes?: number): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function ArtifactCard({ artifact }: Props) {
  const activeID = useCopilotUIStore((s) => s.artifactPanel.activeArtifact?.id);
  const isOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);

  const isActive = isOpen && activeID === artifact.id;
  const classification = classifyArtifact(
    artifact.mimeType,
    artifact.title,
    artifact.sizeBytes,
  );
  const Icon = classification.icon;

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
        className="my-1 flex w-full items-center gap-3 rounded-lg border border-zinc-200 bg-white px-3 py-2.5 text-left transition-colors hover:bg-zinc-50"
      >
        <Icon size={20} className="shrink-0 text-zinc-400" />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-zinc-900">
            {artifact.title}
          </p>
          <p className="text-xs text-zinc-400">
            {classification.label}
            {artifact.sizeBytes
              ? ` \u2022 ${formatSize(artifact.sizeBytes)}`
              : ""}
          </p>
        </div>
        <DownloadSimple size={16} className="shrink-0 text-zinc-400" />
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={() => openArtifact(artifact)}
      className={cn(
        "my-1 flex w-full items-center gap-3 rounded-lg border bg-white px-3 py-2.5 text-left transition-colors hover:bg-zinc-50",
        isActive ? "border-violet-300 bg-violet-50/50" : "border-zinc-200",
      )}
    >
      <Icon
        size={20}
        className={cn(
          "shrink-0",
          isActive ? "text-violet-500" : "text-zinc-400",
        )}
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-zinc-900">
          {artifact.title}
        </p>
        <p className="text-xs text-zinc-400">
          <span
            className={cn(
              "inline-block rounded-full px-1.5 py-0.5 text-xs font-medium",
              artifact.origin === "user-upload"
                ? "bg-blue-50 text-blue-500"
                : "bg-violet-50 text-violet-500",
            )}
          >
            {classification.label}
          </span>
          {artifact.sizeBytes
            ? ` \u2022 ${formatSize(artifact.sizeBytes)}`
            : ""}
        </p>
      </div>
      <CaretRight
        size={16}
        className={cn(
          "shrink-0",
          isActive ? "text-violet-400" : "text-zinc-300",
        )}
      />
    </button>
  );
}
