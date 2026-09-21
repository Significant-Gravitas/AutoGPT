"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { useState } from "react";
import type { ArtifactRef } from "../../../store";
import { useArtifactCard } from "../../ArtifactCard/useArtifactCard";
import type { ArtifactClassification } from "../../ArtifactPanel/helpers";

interface Props {
  artifact: ArtifactRef;
  readOnly?: boolean;
}

const TILE_COLOR: Partial<Record<ArtifactClassification["type"], string>> = {
  pdf: "bg-rose-500",
  csv: "bg-emerald-500",
  json: "bg-sky-500",
  code: "bg-sky-500",
  html: "bg-sky-500",
  react: "bg-sky-500",
  markdown: "bg-amber-500",
  text: "bg-amber-500",
  video: "bg-violet-500",
  image: "bg-violet-500",
};

export function AttachmentPreview({ artifact, readOnly }: Props) {
  const { isActive, classification, handleOpen } = useArtifactCard(
    artifact,
    readOnly,
  );
  const [imageFailed, setImageFailed] = useState(false);
  const showImage = classification.type === "image" && !imageFailed;

  if (showImage) {
    return (
      <button
        type="button"
        onClick={handleOpen}
        aria-label={`Open ${artifact.title}`}
        data-testid="attachment-preview-image"
        className={cn(
          "overflow-hidden rounded-2xl border bg-zinc-50 transition-colors animate-in fade-in slide-in-from-bottom-2 fill-mode-both [animation-duration:500ms] hover:border-zinc-300",
          isActive ? "border-violet-300" : "border-zinc-200",
        )}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={artifact.sourceUrl}
          alt={artifact.title}
          loading="lazy"
          onError={() => setImageFailed(true)}
          className="block max-h-64 max-w-[18rem] object-cover"
        />
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={handleOpen}
      data-testid="attachment-preview-file"
      className={cn(
        "flex min-w-0 max-w-[18rem] items-center gap-3 rounded-2xl border bg-white py-2 pl-2 pr-4 text-left transition-colors animate-in fade-in slide-in-from-bottom-2 fill-mode-both [animation-duration:500ms] hover:bg-zinc-50",
        isActive ? "border-violet-300 bg-violet-50/50" : "border-zinc-200",
      )}
    >
      <span
        className={cn(
          "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-white",
          TILE_COLOR[classification.type] ?? "bg-zinc-400",
        )}
      >
        <Icon icon={classification.icon} size={20} />
      </span>
      <span className="min-w-0">
        <span className="block truncate text-sm font-medium text-zinc-900">
          {artifact.title}
        </span>
        <span className="block text-xs text-zinc-500">
          {classification.label}
        </span>
      </span>
    </button>
  );
}
