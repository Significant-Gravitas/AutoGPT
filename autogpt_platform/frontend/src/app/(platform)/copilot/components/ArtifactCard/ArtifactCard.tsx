"use client";

import { cn } from "@/lib/utils";
import type { ArtifactRef } from "../../store";
import { ArrowRight01Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useArtifactCard } from "./useArtifactCard";

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
  const { isActive, classification, handleOpen, handleDownloadOnly } =
    useArtifactCard(artifact, readOnly);

  if (!classification.openable) {
    return (
      <button
        type="button"
        onClick={handleDownloadOnly}
        className="my-1 flex w-full min-w-0 items-center gap-3 rounded-2xl border border-zinc-200 bg-white px-3 py-2.5 text-left transition-colors animate-in fade-in slide-in-from-bottom-2 fill-mode-both [animation-duration:500ms] hover:bg-zinc-50"
      >
        <Icon
          icon={classification.icon}
          size={20}
          className="shrink-0 text-zinc-400"
        />
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
        <Icon
          icon={Download01Icon}
          size={16}
          className="shrink-0 text-zinc-400"
        />
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={handleOpen}
      className={cn(
        "my-1 flex w-full min-w-0 items-center gap-3 rounded-2xl border bg-white px-3 py-2.5 text-left transition-colors animate-in fade-in slide-in-from-bottom-2 fill-mode-both [animation-duration:500ms] hover:bg-zinc-50",
        isActive ? "border-violet-300 bg-violet-50/50" : "border-zinc-200",
      )}
    >
      <Icon
        icon={classification.icon}
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
      <Icon
        icon={ArrowRight01Icon}
        size={16}
        className={cn(
          "shrink-0",
          isActive ? "text-violet-400" : "text-zinc-300",
        )}
      />
    </button>
  );
}
